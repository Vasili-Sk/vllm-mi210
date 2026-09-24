# SPDX-License-Identifier: Apache-2.0
"""Disk-backed (read-only mmap) PLE n-gram table for the AMD/ROCm path.

Motivation
----------
``Qwen3.8-Flash-Next`` ships the PLE n-gram table as FP8 E4M3 split row-wise into
``split_ngram_parts`` shards. For this checkpoint that is 128 shards of
``[2500012, 160]`` == 51.2 GB. The device path allocates it in VRAM and the UVA
offload path (``Qwen4ExpPLEPinnedHostEmbedding``) allocates it in *pinned* host
memory. Pinned memory is neither swappable nor reclaimable, so on a machine that
already holds most of RAM (for example an iGPU using GTT for its KV cache) the
pinned table simply does not fit.

This module keeps the table where llama.cpp keeps it (``TENSOR_READ_LAZY`` in
``src/llama-model-loader.h``): as clean, file-backed, evictable page cache. The
registered ``weight`` parameter is a zero-element placeholder, so the table is
never materialised in RAM or VRAM. Lookups gather rows from a read-only ``mmap``.

The gather is genuinely small. Per token the layer touches ``ngram_heads`` rows
of ``embedding_dim`` bytes -- 16 x 160 B = 2.56 KB for this checkpoint. Resident
memory is therefore bounded by the distinct n-gram rows a workload visits, not
by the 51.2 GB table size.
"""

import mmap
import os
import json
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.logger import init_logger

from .ngram_embedding import Qwen4ExpPLEEmbedding

logger = init_logger(__name__)

_GRANULARITY = mmap.ALLOCATIONGRANULARITY


class _ShardView:
    """One checkpoint shard, mapped read-only and viewed as a row matrix."""

    __slots__ = ("path", "rows", "row_bytes", "_mm", "_array")

    def __init__(self, path: str, tensor_name: str, row_bytes: int) -> None:
        self.path = path
        self.row_bytes = row_bytes
        # Keep the descriptor open until after mmap() has dup'd it. safetensors
        # stores [u64 header length][header JSON][tensor blobs].
        handle = open(path, "rb")
        header_size = int.from_bytes(handle.read(8), byteorder="little")
        header = json.loads(handle.read(header_size))
        info = header[tensor_name]
        # safetensors data_offsets are relative to the end of the header, not
        # to the start of the file.
        data_start = 8 + header_size
        begin = data_start + int(info["data_offsets"][0])
        end = data_start + int(info["data_offsets"][1])
        nbytes = end - begin
        if nbytes % row_bytes:
            raise ValueError(
                f"Shard {path} tensor {tensor_name} spans {nbytes} bytes, "
                f"which is not a multiple of row size {row_bytes}"
            )
        self.rows = nbytes // row_bytes
        # mmap offsets must be page aligned. Map from the rounded-down offset and
        # trim the remainder out of the resulting array.
        # mmap offsets must be page aligned. Map from the rounded-down offset,
        # clamp to end of file, then slice the remainder off explicitly.
        pad = begin % _GRANULARITY
        start = begin - pad
        file_size = os.fstat(handle.fileno()).st_size
        map_len = min(nbytes + pad, file_size - start)
        if map_len < nbytes + pad:
            raise ValueError(
                f"Shard {path}: tensor {tensor_name} spans past end of file "
                f"(start={start:,}, need={nbytes + pad:,}, file={file_size:,})"
            )
        try:
            self._mm = mmap.mmap(
                handle.fileno(),
                map_len,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_SHARED,
                offset=start,
            )
        finally:
            handle.close()
        raw = np.frombuffer(self._mm, dtype=np.uint8, count=nbytes + pad)
        self._array = raw[pad : pad + nbytes].reshape(self.rows, row_bytes)

    def gather(self, local_rows: np.ndarray, out: np.ndarray, out_pos) -> None:
        """Copy the requested rows into ``out``. Only touched pages fault in.

        Assignment must be in place. ``np.copyto(out[out_pos], ...)`` would write
        into the temporary that fancy indexing returns and silently lose the data.
        """
        out[out_pos] = self._array[local_rows]

    def advise_dontneed(self) -> None:
        """Ask the kernel to drop this shard from page cache after a bulk read."""
        try:
            os.posix_fadvise(
                self._mm.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
            )
        except (AttributeError, OSError):
            pass

    def close(self) -> None:
        try:
            self._mm.close()
        except (BufferError, OSError):
            pass


class PleMmapTable:
    """Row gather over a vocab-sharded safetensors table, with no residency.

    Row ``i`` of the logical table lives at shard ``i // rows_per_shard``, local
    row ``i % rows_per_shard``. That is exactly the mapping vLLM's own PLE loader
    uses (``checkpoint_start = shard_index * shard_size``), so both paths agree.
    """

    def __init__(
        self,
        model_dir: str,
        shard_tensor_names: dict[int, str],
        rows_per_shard: int,
        embedding_dim: int,
        row_bytes: int,
        num_rows: int,
    ) -> None:
        self.rows_per_shard = rows_per_shard
        self.embedding_dim = embedding_dim
        self.row_bytes = row_bytes
        self.num_rows = num_rows
        self._views: dict[int, _ShardView] = {}
        for index in sorted(shard_tensor_names):
            self._views[index] = _ShardView(
                shard_tensor_names[index][0],
                shard_tensor_names[index][1],
                row_bytes,
            )
        logger.info(
            "PLE mmap table ready: %d shards, %d rows/shard, %d cols, "
            "%.2f GB on disk, resident cost ~0 (clean page cache)",
            len(self._views),
            rows_per_shard,
            embedding_dim,
            len(self._views) * rows_per_shard * row_bytes / 1e9,
        )

    def _fill(self, flat: np.ndarray, out: np.ndarray) -> None:
        """Fill ``out`` (rows x row_bytes) by reading only the touched pages."""
        valid = (flat >= 0) & (flat < self.num_rows)
        if flat.size:
            # Bucket row ids by shard so each mapped file is touched once.
            shard_of = np.where(valid, flat // self.rows_per_shard, -1)
            for shard_index in np.unique(shard_of[valid]):
                view = self._views[int(shard_index)]
                selected = np.where(shard_of == shard_index)[0]
                local = (flat[selected] - int(shard_index) * self.rows_per_shard)
                view.gather(local, out, selected)
        if flat.size and not valid.all():
            # Rows outside the trained vocabulary read as zero, matching the
            # in-range mask of the resident/pinned lookups.
            out[~valid] = 0

    def _to_rows(self, raw: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Reinterpret gathered bytes as the checkpoint storage dtype."""
        if self.row_bytes % self.embedding_dim:
            raise ValueError(
                f"row_bytes={self.row_bytes} is not a multiple of "
                f"embedding_dim={self.embedding_dim}"
            )
        rows_per_dtype = self.row_bytes // self.embedding_dim
        if rows_per_dtype != 1:
            raise NotImplementedError(
                "PLE mmap gather assumes one byte per element (FP8 E4M3); "
                f"got row_bytes={self.row_bytes}, "
                f"embedding_dim={self.embedding_dim}"
            )
        return raw.view(torch.float8_e4m3fn).view(
            *original_shape, self.embedding_dim
        )

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather rows into ordinary memory. Used by tests and warmup."""
        original_shape = ids.shape
        flat = ids.reshape(-1).to("cpu", dtype=torch.int64).numpy()
        out = np.empty((flat.size, self.row_bytes), dtype=np.uint8)
        self._fill(flat, out)
        return self._to_rows(torch.from_numpy(out), original_shape)

    def gather_pinned(self, ids: torch.Tensor, stage_fn) -> torch.Tensor:
        """Gather rows into pinned staging storage, then copy to the device.

        CUDA graph capture rejects CPU to GPU copies unless the source is
        pinned, so gathered bytes land in a small pinned buffer first. Only the
        rows a request touches are ever resident -- a few KB per token -- while
        the 51.2 GB table stays on disk as reclaimable page cache.
        """
        original_shape = ids.shape
        flat = ids.reshape(-1).to("cpu", dtype=torch.int64).numpy()
        stage = stage_fn(flat.size)
        self._fill(flat, stage.numpy())
        on_device = stage.to(device=ids.device, non_blocking=False)
        return self._to_rows(on_device.clone(), original_shape)

    def gather_into(self, ids: torch.Tensor, output: torch.Tensor, stage_fn) -> None:
        """Gather mmap rows into a stable device buffer before model forward."""
        flat = ids.reshape(-1).to("cpu", dtype=torch.int64).numpy()
        stage = stage_fn(flat.size)
        self._fill(flat, stage.numpy())
        host_rows = self._to_rows(stage, ids.shape)
        if tuple(output.shape) != tuple(host_rows.shape):
            raise ValueError(
                f"PLE mmap staging output has {tuple(output.shape)}, "
                f"expected {tuple(host_rows.shape)}"
            )
        output.copy_(host_rows, non_blocking=False)

    def advise_dontneed(self) -> None:
        for view in self._views.values():
            view.advise_dontneed()


class Qwen4ExpPLEMmapEmbedding(Qwen4ExpPLEEmbedding):
    """PLE embedding whose table stays on disk and is gathered per lookup."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Any,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        mmap_table: PleMmapTable | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        # The table is attached after the shards are discovered by the layer.
        self._mmap_table: PleMmapTable | None = mmap_table
        # Pinned staging buffer for gathered rows, reused across calls. Sized
        # from the scheduler's token budget so it never has to grow during CUDA
        # graph capture, where a reallocation or an unpinned copy is illegal.
        self._pinned: torch.Tensor | None = None
        self._pinned_rows = 0
        self._wanted_rows = max(1, int(max_total_tokens)) * max(
            1, int(num_ngram_heads)
        )

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Register a shape-correct but memory-free placeholder weight.

        Returning ``torch.empty(0, ...)`` keeps the parameter addressable for
        vLLM's module bookkeeping while allocating none of the table. The real
        rows are read from disk in :meth:`_lookup`.
        """
        del num_embeddings, embedding_dim
        return torch.empty(0, dtype=dtype)

    def attach_mmap_table(self, table: PleMmapTable) -> None:
        self._mmap_table = table

    @property
    def mmap_table(self) -> PleMmapTable | None:
        return self._mmap_table

    def _stage_for(self, rows: int) -> torch.Tensor:
        """Return a pinned uint8 buffer holding at least ``rows`` rows."""
        table = self._mmap_table
        assert table is not None
        want = max(int(rows), self._wanted_rows)
        if self._pinned is None or self._pinned_rows < want:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PLE mmap staging buffer would have to grow during CUDA "
                    f"graph capture ({self._pinned_rows} -> {want} rows). "
                    "max_total_tokens is too small for this batch."
                )
            self._pinned = torch.empty(
                want, table.row_bytes, dtype=torch.uint8, pin_memory=True
            )
            self._pinned_rows = want
            logger.info(
                "PLE mmap staging buffer: %d rows x %d B = %.1f MiB pinned",
                want,
                table.row_bytes,
                want * table.row_bytes / 1048576,
            )
        return self._pinned[:rows]

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather rows from disk through pinned staging, keeping storage dtype."""
        if self._mmap_table is None:
            raise RuntimeError(
                "PLE mmap embedding was used before its table was attached. "
                "The layer must attach it during weight loading."
            )
        gathered = self._mmap_table.gather_pinned(
            input_ids, self._stage_for
        )
        if output is None:
            return gathered
        if tuple(output.shape) != tuple(gathered.shape):
            raise ValueError(
                f"PLE mmap gather produced {tuple(gathered.shape)}, "
                f"but output is {tuple(output.shape)}"
            )
        output.copy_(gathered)
        return output

    def gather_into(self, ngram_ids: torch.Tensor, output: torch.Tensor) -> None:
        """Fill stable GPU staging storage outside compiled model execution."""
        if self._mmap_table is None:
            raise RuntimeError("PLE mmap table is not attached")
        self._mmap_table.gather_into(ngram_ids, output, self._stage_for)

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous disk lookup, matching the pinned-host entry point."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        embeddings = self._reduce_etp_embeddings(embeddings)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Prefetch is unused; the custom-op escape calls sync_lookup."""
        return None

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

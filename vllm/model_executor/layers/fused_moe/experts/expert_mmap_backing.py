# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable safetensors backing for WNA16 expert-row refill."""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


class ExpertMmapBacking:
    """Repack one GPTQ expert into existing WNA16 runtime tensors."""

    def __init__(self, model_dir: str | Path, cache_mib: int = 1024) -> None:
        self.root = Path(model_dir)
        index_path = self.root / "model.safetensors.index.json"
        with index_path.open() as index_file:
            self.weight_map = json.load(index_file)["weight_map"]
        self.cache_bytes = max(0, cache_mib) * 2**20
        self.cached_bytes = 0
        self.lru: OrderedDict[tuple[int, int], list[tuple[str, int, int]]] = (
            OrderedDict()
        )
        self.contexts: dict[str, Any] = {}
        self.handles: dict[str, Any] = {}
        self.raw_files: dict[str, Any] = {}
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}

    @staticmethod
    def _key(layer: int, expert: int, projection: str, tensor: str) -> str:
        return (
            f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            f"{projection}.{tensor}"
        )

    def _handle(self, filename: str):
        handle = self.handles.get(filename)
        if handle is None:
            context = safe_open(str(self.root / filename), framework="pt", device="cpu")
            handle = context.__enter__()
            self.contexts[filename] = context
            self.handles[filename] = handle
        return handle

    def _header(self, filename: str) -> tuple[int, dict[str, Any]]:
        item = self.headers.get(filename)
        if item is not None:
            return item
        raw_file = (self.root / filename).open("rb", buffering=0)
        header_size = struct.unpack("<Q", raw_file.read(8))[0]
        header = json.loads(raw_file.read(header_size))
        item = (8 + header_size, header)
        self.raw_files[filename] = raw_file
        self.headers[filename] = item
        return item

    def _ranges(self, layer: int, expert: int) -> list[tuple[str, int, int]]:
        ranges: list[tuple[str, int, int]] = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for tensor in ("qweight", "scales"):
                name = self._key(layer, expert, projection, tensor)
                filename = self.weight_map[name]
                data_start, header = self._header(filename)
                begin, end = header[name]["data_offsets"]
                ranges.append((filename, data_start + begin, end - begin))
        return ranges

    def _drop_ranges(self, ranges: list[tuple[str, int, int]]) -> None:
        if not hasattr(os, "posix_fadvise"):
            return
        for filename, offset, length in ranges:
            os.posix_fadvise(
                self.raw_files[filename].fileno(),
                offset,
                length,
                os.POSIX_FADV_DONTNEED,
            )

    def _touch_lru(self, layer: int, expert: int) -> None:
        identity = (layer, expert)
        previous = self.lru.pop(identity, None)
        if previous is not None:
            self.cached_bytes -= sum(length for _, _, length in previous)
        ranges = self._ranges(layer, expert)
        self.lru[identity] = ranges
        self.cached_bytes += sum(length for _, _, length in ranges)
        while self.lru and self.cached_bytes > self.cache_bytes:
            _, old_ranges = self.lru.popitem(last=False)
            self.cached_bytes -= sum(length for _, _, length in old_ranges)
            self._drop_ranges(old_ranges)

    def _tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        return self._handle(filename).get_tensor(name)

    def refill(
        self,
        layer: int,
        expert: int,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> None:
        """Overwrite one cold slot with the exact Triton WNA16 layout."""
        gate_q = self._tensor(self._key(layer, expert, "gate_proj", "qweight"))
        up_q = self._tensor(self._key(layer, expert, "up_proj", "qweight"))
        down_q = self._tensor(self._key(layer, expert, "down_proj", "qweight"))
        gate_s = self._tensor(self._key(layer, expert, "gate_proj", "scales"))
        up_s = self._tensor(self._key(layer, expert, "up_proj", "scales"))
        down_s = self._tensor(self._key(layer, expert, "down_proj", "scales"))

        split = gate_q.shape[1]
        w1_i32 = w1.view(torch.int32)
        w2_i32 = w2.view(torch.int32)
        w1_i32[:split].copy_(gate_q.T)
        w1_i32[split:].copy_(up_q.T)
        w2_i32.copy_(down_q.T)
        s1[:split].copy_(gate_s.T)
        s1[split:].copy_(up_s.T)
        s2.copy_(down_s.T)
        self._touch_lru(layer, expert)

    def close(self) -> None:
        for filename, context in list(self.contexts.items()):
            context.__exit__(None, None, None)
            self.handles.pop(filename, None)
        self.contexts.clear()
        for raw_file in self.raw_files.values():
            raw_file.close()
        self.raw_files.clear()
        self.headers.clear()
        self.lru.clear()
        self.cached_bytes = 0

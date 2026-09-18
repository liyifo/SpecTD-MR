import json
import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _load_raw(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.json'}:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if ext in {'.pkl', '.pickle', '.dill'}:
        with open(path, 'rb') as f:
            return pickle.load(f)
    if ext in {'.pt', '.pth'}:
        return torch.load(path, map_location='cpu')
    raise ValueError(f'Unsupported hierarchy file: {path}')


def _normalize_path(raw_value) -> Optional[Tuple[str, ...]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        return tuple(str(v) for v in raw_value if v != '')
    if isinstance(raw_value, str):
        for sep in ('>', '/', '|', ','):
            raw_value = raw_value.replace(sep, ' ')
        tokens = [tok for tok in raw_value.split() if tok]
        return tuple(tokens)
    if isinstance(raw_value, dict):
        if 'path' in raw_value:
            return _normalize_path(raw_value['path'])
        if 'levels' in raw_value:
            return _normalize_path(raw_value['levels'])
    return None


class HierarchyEncoder:
    def __init__(self, path: str):
        raw = _load_raw(path)
        if not isinstance(raw, dict):
            raise ValueError('Hierarchy file must contain a mapping from code to path information.')
        self.paths: Dict[str, Tuple[str, ...]] = {}
        for key, value in raw.items():
            path = _normalize_path(value)
            if path:
                self.paths[str(key)] = path

    def get_path(self, code: str) -> Optional[Tuple[str, ...]]:
        return self.paths.get(str(code))

    def distance(self, code_a: str, code_b: str) -> Optional[int]:
        pa = self.get_path(code_a)
        pb = self.get_path(code_b)
        if pa is None or pb is None:
            return None
        common = 0
        for va, vb in zip(pa, pb):
            if va == vb:
                common += 1
            else:
                break
        return len(pa) + len(pb) - 2 * common

    def average_distance(self, code: str, peers: Iterable[str]) -> Optional[float]:
        distances = []
        for peer in peers:
            dist = self.distance(code, peer)
            if dist is not None:
                distances.append(dist)
        if not distances:
            return None
        return float(sum(distances) / len(distances))


class RelativePositionBucketizer:
    def __init__(self, boundaries: Sequence[int]):
        if not boundaries:
            raise ValueError('boundaries must be non-empty')
        self.boundaries = sorted(set(boundaries))
        self.max_bucket = len(self.boundaries)
        self.unknown_bucket = self.max_bucket + 1

    def __call__(self, distance: Optional[float]) -> int:
        if distance is None:
            return self.unknown_bucket
        for idx, bound in enumerate(self.boundaries):
            if distance <= bound:
                return idx
        return self.max_bucket


def build_edge_relative_positions(visit_summaries,
                                  membership_lookup: Dict[Tuple[int, int], int],
                                  node_codes: Sequence[str],
                                  hierarchy: HierarchyEncoder,
                                  num_edges: int,
                                  bucketizer: RelativePositionBucketizer) -> torch.Tensor:
    rel_pos = torch.full((num_edges,), bucketizer.unknown_bucket, dtype=torch.long)

    def _fill(nodes: List[int], visit_node_id: int):
        if not nodes:
            return
        codes = [node_codes[n] for n in nodes]
        for idx, node_id in enumerate(nodes):
            peers = codes[:idx] + codes[idx+1:]
            avg_dist = hierarchy.average_distance(codes[idx], peers)
            bucket = bucketizer(avg_dist)
            edge_idx = membership_lookup.get((node_id, visit_node_id))
            if edge_idx is not None:
                rel_pos[edge_idx] = bucket

    for visit in visit_summaries:
        _fill(visit.diag_nodes, visit.visit_node_id)
        _fill(visit.proc_nodes, visit.visit_node_id)
        _fill(visit.med_nodes, visit.visit_node_id)

    return rel_pos

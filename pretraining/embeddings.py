import json
import os
import pickle
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


def _load_raw_resource(path: Optional[str]):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f'Embedding file not found: {path}')
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.pt', '.pth'}:
        return torch.load(path, map_location='cpu')
    if ext in {'.pkl', '.pickle', '.dill'}:
        with open(path, 'rb') as f:
            return pickle.load(f)
    if ext in {'.json'}:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if ext in {'.npy'}:
        return np.load(path, allow_pickle=True)
    if ext in {'.npz'}:
        npz = np.load(path, allow_pickle=True)
        if 'embeddings' in npz and 'ids' in npz:
            return {'embeddings': npz['embeddings'], 'ids': npz['ids']}
        return {k: npz[k] for k in npz.files}
    raise ValueError(f'Unsupported embedding format: {path}')


def _ensure_tensor(vec: TensorLike) -> torch.Tensor:
    if isinstance(vec, torch.Tensor):
        return vec.float()
    if isinstance(vec, np.ndarray):
        return torch.from_numpy(vec).float()
    if isinstance(vec, (list, tuple)):
        return torch.tensor(vec, dtype=torch.float32)
    raise TypeError(f'Unsupported vector type: {type(vec)}')


def _normalize_resource(raw) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if 'embeddings' in raw and 'ids' in raw:
            emb = raw['embeddings']
            ids = raw['ids']
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            table = {}
            for idx, key in enumerate(ids):
                table[str(key)] = _ensure_tensor(emb[idx])
            return table
        if 'id2vec' in raw:
            return {str(k): _ensure_tensor(v) for k, v in raw['id2vec'].items()}
        if all(isinstance(v, (torch.Tensor, np.ndarray, list, tuple)) for v in raw.values()):
            return {str(k): _ensure_tensor(v) for k, v in raw.items()}
        raise ValueError('Dictionary embedding file must contain tensors or vectors.')
    if isinstance(raw, (list, tuple)):
        return {str(i): _ensure_tensor(v) for i, v in enumerate(raw)}
    if isinstance(raw, np.ndarray):
        return {str(i): _ensure_tensor(v) for i, v in enumerate(raw)}
    raise TypeError(f'Unsupported embedding container: {type(raw)}')


def load_embedding_table(path: Optional[str]) -> Dict[str, torch.Tensor]:
    if path is None:
        return {}
    print(f'Loading embedding table from: {path}')
    print(f'current working directory: {os.getcwd()}')
    raw = _load_raw_resource(path)
    table = _normalize_resource(raw)
    return table


def as_feature_matrix(ids: Sequence[Union[int, str]],
                      table: Dict[str, torch.Tensor],
                      dim: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    feat_dim = dim
    if feat_dim is None:
        for key in ids:
            tensor = table.get(str(key))
            if tensor is None:
                tensor = table.get(key)
            if tensor is not None:
                feat_dim = tensor.shape[-1]
                break
    if feat_dim is None:
        feat_dim = 1
    features = []
    for key in ids:
        str_key = str(key)
        tensor = table.get(str_key)
        if tensor is None and isinstance(key, str):
            tensor = table.get(key)
        if tensor is None and isinstance(key, int):
            tensor = table.get(str(int(key)))
        if tensor is None:
            tensor = torch.zeros(feat_dim, dtype=torch.float32)
        else:
            tensor = tensor.float()
        if tensor.shape[-1] != feat_dim:
            padded = torch.zeros(feat_dim, dtype=torch.float32)
            take = min(feat_dim, tensor.shape[-1])
            padded[:take] = tensor[:take]
            tensor = padded
        features.append(tensor)
    stacked = torch.stack(features, dim=0)
    return stacked, feat_dim


def merge_or_create(ids: Sequence[Union[int, str]],
                    table: Dict[str, torch.Tensor],
                    target_dim: int) -> torch.Tensor:
    mat, mat_dim = as_feature_matrix(ids, table, target_dim)
    if mat_dim != target_dim:
        if mat_dim < target_dim:
            pad = torch.zeros(mat.size(0), target_dim - mat_dim)
            mat = torch.cat([mat, pad], dim=-1)
        else:
            mat = mat[:, :target_dim]
    return mat


def reduce_with_pca(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        raise ValueError('target_dim must be positive for PCA reduction.')
    original_dim = features.size(1)
    if original_dim <= target_dim:
        return features
    max_rank = min(features.size(0), original_dim)
    eff_dim = min(target_dim, max_rank - 1)
    if eff_dim <= 0:
        return features
    centered = features - features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(centered, q=eff_dim, center=False)
    components = V[:, :target_dim]
    reduced = centered @ components
    return reduced

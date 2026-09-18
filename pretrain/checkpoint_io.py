from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _normalize_safe_value(value: Any) -> Any:
    """Convert NumPy scalar values to types accepted by torch.load's safe mode."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalize_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_safe_value(item) for item in value)
    return value


def save_training_checkpoint(payload: Dict[str, Any], path: Path) -> None:
    """Save a training checkpoint compatible with PyTorch's safe load default."""
    torch.save(_normalize_safe_value(payload), path)

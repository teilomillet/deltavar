import torch
from typing import Dict, List, Union

TensorTree = Union[torch.Tensor, Dict[str, 'TensorTree'], List['TensorTree']]

def flatten_pytree(pytree: TensorTree) -> torch.Tensor:
    """
    Flattens a PyTree (nested dicts/lists of tensors) into a single 1D tensor.
    """
    tensors = []
    if isinstance(pytree, torch.Tensor):
        tensors.append(pytree.reshape(-1))
    elif isinstance(pytree, dict):
        for v in pytree.values():
            tensors.append(flatten_pytree(v))
    elif isinstance(pytree, list):
        for v in pytree:
            tensors.append(flatten_pytree(v))
    else:
        # Handle other potential types like tuples, etc., if necessary
        raise TypeError(f"Unsupported type in PyTree: {type(pytree)}")

    if not tensors:
        return torch.empty(0) # Handle empty pytree
        
    return torch.cat(tensors)

# TODO: Add unflatten_pytree if needed later, requires original structure example

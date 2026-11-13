"""Tensor serialization primitives without training-module dependencies."""
import hashlib
import numpy as np
import torch
from .core import digest

def native_scalars(value):
    """JSON/safe-load compatible scalar metadata, without altering tensor state.

    Preserve containers and integer optimizer keys. RNG arrays are already packed
    explicitly by artifact_rng; this does not convert arrays, dtypes or objects.
    Extended precision/structured scalars have no lossless primitive equivalent.
    """
    if isinstance(value,np.generic):
        result=value.item()
        if type(result) not in (bool,int,float,str):
            raise TypeError(f'Unsupported NumPy metadata scalar {value.dtype}; use explicit tensor/array encoding')
        return result
    if isinstance(value,dict):return {native_scalars(k):native_scalars(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(native_scalars(v) for v in value)
    return value


def tree_cpu(value):
    if isinstance(value,np.generic):return native_scalars(value)
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {native_scalars(k):tree_cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(tree_cpu(v) for v in value)
    return value


def tree_identity(value):
    if isinstance(value,np.generic):return native_scalars(value)
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous()
        return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    if isinstance(value,dict):return {str(k):tree_identity(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tree_identity(v) for v in value]
    return value


def state_identity(model):return digest(tree_identity(model.state_dict()))


def state_schema(state):
    return {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()}

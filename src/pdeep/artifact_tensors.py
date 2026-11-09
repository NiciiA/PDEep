"""Tensor serialization primitives without training-module dependencies."""
import hashlib
import torch
from .core import digest


def tree_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:tree_cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(tree_cpu(v) for v in value)
    return value


def tree_identity(value):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous()
        return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    if isinstance(value,dict):return {str(k):tree_identity(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tree_identity(v) for v in value]
    return value


def state_identity(model):return digest(tree_identity(model.state_dict()))


def state_schema(state):
    return {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()}

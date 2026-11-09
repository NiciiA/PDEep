"""Factories for independently owned representations; no A/B/C predictors."""
import torch
from torch import nn
from dataclasses import fields

from .core import require
from .diagnostics import RandomStreams
from .encoding import EquationEncoder,EquationVocabulary,ObservationEncoder,ConditionalQueryEncoder
from .memory import CoefficientEncoder,SpatialMemory,MemoryRetriever
from .runtime import resolve_device


def float32_batch(batch,*,device="cpu"):
    """Explicit model-input COPY/transfer; original reference batch stays untouched."""
    resolved,_ = resolve_device(device,"float32")
    return type(batch)(**{f.name:getattr(batch,f.name).to(device=resolved,dtype=torch.float32
                          if getattr(batch,f.name).is_floating_point() else getattr(batch,f.name).dtype)
                          for f in fields(batch)})


class RepresentationModules(nn.Module):
    """A module inventory, not an end-to-end model: call each encoder explicitly."""
    def __init__(self,owner,normalizers,vocabulary,*,spatial_memory,geometry_provider=None):
        super().__init__()
        self.owner = owner
        self.equation = EquationEncoder(vocabulary)
        self.observation = ObservationEncoder(normalizers)
        self.query = ConditionalQueryEncoder()
        self.coefficient = CoefficientEncoder()
        self.memory = SpatialMemory(geometry_provider=geometry_provider) if spatial_memory else None
        self.retrieval = MemoryRetriever() if spatial_memory else None


def build_representations(owner,normalizers,vocabulary,*,root_seed=20261108,spatial_memory=None,geometry_provider=None):
    require(owner in {"A","B","C"},"representation owner must be A, B or C")
    spatial_memory = owner == "A" if spatial_memory is None else spatial_memory
    require(type(spatial_memory) is bool,"spatial_memory must be bool")
    # Preserve global CPU RNG and avoid seeding or touching accelerator generators.
    with torch.random.fork_rng(devices=[]):
        seed_state = RandomStreams(root_seed).torch("model_initialization",owner,"representations-v1").get_state()
        torch.random.set_rng_state(seed_state)
        return RepresentationModules(owner,normalizers,vocabulary,spatial_memory=spatial_memory,geometry_provider=geometry_provider)


def parameter_inventory(modules):
    return {name:sum(p.numel() for p in module.parameters()) if module is not None else 0
            for name,module in (("equation",modules.equation),("observation",modules.observation),
                                ("query",modules.query),("coefficient",modules.coefficient),
                                ("memory",modules.memory),("retrieval",modules.retrieval))}


def computational_estimate(*,batch=3,nodes=15,edges=16,observations=64,queries=128):
    """Float32 tensor-storage estimates; exclude autograd, allocator and workspace overhead."""
    return {"width":96,"heads":4,"layers":2,"anchors":80,
            "graph_nodes_bytes":batch*nodes*96*4,"observation_tokens_bytes":batch*observations*96*4,
            "anchor_tokens_bytes":batch*80*96*4,"saved_attention_bytes":batch*2*4*80*observations*4,
            "query_features_bytes":batch*queries*96*4,"B_coefficient_tokens_bytes":batch*18*96*4,
            "retrieval_gaussian_weights_bytes":batch*queries*80*4,
            "retrieval_scale_contexts_bytes":batch*queries*2*96*4,
            "cross_attention_dot_product_MACs":batch*2*2*80*observations*96,
            "cross_attention_logits_elements":batch*2*4*80*observations,
            "gaussian_distance_pairs":batch*queries*80,"retrieval_weighted_pool_MACs":batch*queries*80*96,
            "graph_message_edges_bidirectional_two_layers":batch*edges*2*2,
            "complexity":"graph O(B E W^2); cross-attention O(B L (M+N) W^2 + B L M N W); retrieval O(B Q M W + B Q W^2)",
            "excludes":"MLP/projection activations, positional-bias activations, gradients, parameter/optimizer state and allocator overhead"}

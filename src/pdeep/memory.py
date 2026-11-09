"""Optional spatial representations and coefficient tokens, separate from physics."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .batches import masked_values
from .core import require
from .encoding import (WIDTH,CoordinateFeatures,FourierFeatures,boundary_distances,
                       masked_pool,module_input)
from .geometry import geometry_for


def anchor_geometry(provider=None):
    """Compatibility API; placement is delegated to the geometry provider."""
    layout = (geometry_for() if provider is None else provider).anchor_layout().validate()
    return layout.positions,layout.levels


def masked_softmax(logits,valid):
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    weights = torch.softmax(torch.where(valid,logits,torch.full_like(logits,-torch.finfo(logits.dtype).max)),dim=-1)
    weights = torch.where(valid,weights,torch.zeros_like(weights))
    return weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


class ObservationMemoryLayer(nn.Module):
    heads = 4

    def __init__(self):
        super().__init__()
        self.q,self.k,self.v = (nn.Linear(WIDTH,WIDTH) for _ in range(3))
        self.output = nn.Linear(WIDTH,WIDTH)
        # dx,dy, distance², optional physical time, presence, quantity, log noise
        self.position_bias = nn.Sequential(nn.Linear(7,32),nn.SiLU(),nn.Linear(32,self.heads))
        self.norm = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH,2*WIDTH),nn.SiLU(),nn.Linear(2*WIDTH,WIDTH))
        self.ff_norm = nn.LayerNorm(WIDTH)
        self.coarse_to_fine,self.fine_to_coarse = nn.Linear(WIDTH,WIDTH),nn.Linear(WIDTH,WIDTH)
        self.mix_norm = nn.LayerNorm(WIDTH)

    def forward(self,memory,observations,anchors,levels):
        b,m,w = memory.shape
        n = observations.tokens.shape[1]
        h,d = self.heads,w//self.heads
        q = self.q(memory).reshape(b,m,h,d).transpose(1,2)
        k = self.k(observations.tokens).reshape(b,n,h,d).transpose(1,2)
        v = self.v(observations.tokens).reshape(b,n,h,d).transpose(1,2)
        relative = observations.position[:,None]-anchors[None,:,None]
        t = masked_values(observations.physical_time,observations.time_present)
        scalar = torch.cat((t,observations.time_present[...,None].to(memory.dtype),
                            observations.quantity_id[...,None].to(memory.dtype),observations.log_noise),dim=-1)
        bias_input = torch.cat((relative,relative.square().sum(-1,keepdim=True),scalar[:,None].expand(-1,m,-1,-1)),dim=-1)
        bias = self.position_bias(bias_input).permute(0,3,1,2)
        weights = masked_softmax(q@k.transpose(-1,-2)/math.sqrt(d)+bias,observations.valid[:,None,None,:])
        update = self.output((weights@v).transpose(1,2).reshape(b,m,w))
        update = torch.where(observations.valid.any(1)[:,None,None],update,torch.zeros_like(update))
        memory = self.norm(memory+update)
        memory = self.ff_norm(memory+self.ff(memory))
        coarse,fine = memory[:,levels==0].mean(1),memory[:,levels==1].mean(1)
        # Cross-scale pooled context, without periodic adjacency or a dense 80² graph.
        mixed = torch.where((levels==0)[None,:,None],self.fine_to_coarse(fine)[:,None],
                             self.coarse_to_fine(coarse)[:,None])
        return self.mix_norm(memory+mixed),weights


@dataclass(frozen=True)
class MemoryState:
    tokens: torch.Tensor  # [B,80,96]
    global_context: torch.Tensor  # [B,96], constructed once, query-chunk independent
    positions: torch.Tensor  # [80,2]
    levels: torch.Tensor  # [80], 0 coarse, 1 fine
    attention: torch.Tensor | None  # optional [B,2,4,80,N]


class SpatialMemory(nn.Module):
    def __init__(self,geometry_provider=None):
        super().__init__()
        self.domain_geometry = geometry_for() if geometry_provider is None else geometry_provider
        positions,levels = anchor_geometry(self.domain_geometry)
        self.register_buffer("positions",positions)
        self.register_buffer("levels",levels)
        self.fourier = FourierFeatures()
        self.geometry = nn.Sequential(nn.Linear(2+self.domain_geometry.boundary_feature_size+1+24+2,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH))
        self.equation = nn.Linear(WIDTH,WIDTH)
        self.known = nn.Linear(WIDTH,WIDTH,bias=False)
        self.known_presence = nn.Embedding(2,WIDTH)
        self.initial_norm = nn.LayerNorm(WIDTH)
        self.layers = nn.ModuleList([ObservationMemoryLayer(),ObservationMemoryLayer()])
        self.global_projection = nn.Linear(3*WIDTH,WIDTH)

    def forward(self,observations,equation_context,*,known_context=None,known_valid=None,return_attention=False):
        module_input(self,observations.tokens,"observation tokens")
        b = len(observations.tokens)
        require(equation_context.shape == (b,WIDTH), "memory equation context shape mismatch")
        module_input(self,equation_context,"memory equation context")
        distances = self.domain_geometry.boundary_features(self.positions)
        geometry = torch.cat((self.positions,distances,distances.min(-1,keepdim=True).values,
                              self.fourier(self.positions),F.one_hot(self.levels,2).to(self.positions.dtype)),dim=-1)
        memory = self.geometry(geometry)[None].expand(b,-1,-1)+self.equation(equation_context)[:,None]
        if known_context is None:
            require(known_valid is None,"known-field mask supplied without field context")
            known_valid = torch.zeros(b,len(self.positions),dtype=torch.bool,device=memory.device)
            known_context = torch.zeros_like(memory)
        else:
            require(known_context.shape == (b,len(self.positions),WIDTH) and known_valid is not None
                    and known_valid.shape == (b,len(self.positions)) and known_valid.dtype == torch.bool,
                    "known fields require [B,M,96] context and [B,M] bool mask")
            module_input(self,known_context,"known field context")
            require(bool(torch.isfinite(known_context[known_valid]).all()), "nonfinite known field context")
        memory = self.initial_norm(memory+self.known(masked_values(known_context,known_valid))+
                                   self.known_presence(known_valid.long()))
        attentions = []
        for layer in self.layers:
            memory,attention = layer(memory,observations,self.positions,self.levels)
            if return_attention:
                attentions.append(attention)
        global_context = self.global_projection(torch.cat((equation_context,observations.summary,memory.mean(1)),dim=-1))
        return MemoryState(memory,global_context,self.positions,self.levels,
                           torch.stack(attentions,dim=1) if return_attention else None)


@dataclass(frozen=True)
class Retrieval:
    features: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    coarse_context: torch.Tensor  # exact Gaussian m_coarse(x), [B,Q,96]
    fine_context: torch.Tensor


def gaussian_readout(position,anchors,tokens,lengthscale):
    """All-anchor normalized Gaussian weights and weighted feature mean.

    Low-level float32/float64 function for independent derivative verification.
    No hard selection, content logits or learned bandwidth modifies the kernel.
    """
    require(position.ndim==3 and position.shape[-1]==2 and anchors.ndim==2 and anchors.shape[1]==2,
            "Gaussian positions must be [B,Q,2] / [M,2]")
    require(tokens.ndim==3 and tokens.shape[:2]==(len(position),len(anchors)) and len(anchors)>0,"Gaussian token shape mismatch")
    require(position.dtype in (torch.float32,torch.float64) and position.dtype==anchors.dtype==tokens.dtype
            and position.device==anchors.device==tokens.device,"Gaussian dtype/device mismatch")
    require(bool(torch.isfinite(position).all()) and bool(torch.isfinite(anchors).all()) and bool(torch.isfinite(tokens).all()),
            "nonfinite Gaussian inputs")
    require(bool(torch.isfinite(torch.as_tensor(lengthscale)).all()) and bool(lengthscale>0),"positive finite Gaussian lengthscale required")
    distance=(position[:,:,None]-anchors[None,None]).square().sum(-1)
    weights=torch.softmax(-distance/(2*lengthscale**2),dim=-1)
    return weights@tokens,weights


class MemoryRetriever(nn.Module):
    def __init__(self,*,coarse_lengthscale=.30,fine_lengthscale=.15):
        super().__init__()
        require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (coarse_lengthscale,fine_lengthscale)),
                "Gaussian lengthscales must be positive finite numbers")
        self.register_buffer("lengthscales",torch.tensor((coarse_lengthscale,fine_lengthscale),dtype=torch.float32))
        self.output = nn.Sequential(nn.Linear(4*WIDTH,WIDTH),nn.SiLU(),nn.LayerNorm(WIDTH))

    def forward(self,state,query,query_features,specs):
        query.validate(specs)
        module_input(self,query.position,"retrieval queries")
        require(query_features.shape == (*query.valid.shape,WIDTH),"query feature shape mismatch")
        q = query.masked()
        module_input(self,query_features,"retrieval features")
        query_features = masked_values(query_features,q.valid)
        coarse_ids = torch.nonzero(state.levels==0,as_tuple=True)[0]
        fine_ids = torch.nonzero(state.levels==1,as_tuple=True)[0]
        coarse,cw = gaussian_readout(q.position,state.positions[coarse_ids],state.tokens[:,coarse_ids],self.lengthscales[0])
        fine,fw = gaussian_readout(q.position,state.positions[fine_ids],state.tokens[:,fine_ids],self.lengthscales[1])
        b,n = q.valid.shape
        indices = torch.cat((coarse_ids,fine_ids))[None,None].expand(b,n,-1)
        weights = torch.cat((cw,fw),dim=-1)  # each scale sums to one; total is two
        global_context = state.global_context[:,None].expand(-1,n,-1)
        features = self.output(torch.cat((query_features,coarse,fine,global_context),dim=-1))
        return Retrieval(masked_values(features,q.valid),masked_values(indices,q.valid),masked_values(weights,q.valid),
                         masked_values(coarse,q.valid),masked_values(fine,q.valid))


@dataclass(frozen=True)
class CoefficientTokenBatch:
    xi: torch.Tensor
    active: torch.Tensor
    modes: torch.Tensor  # [B,18,2], exact (m,n) identity
    block: torch.Tensor  # 0=a, 1=b
    prior_scale: torch.Tensor
    gradient: torch.Tensor
    previous_update: torch.Tensor


def coefficient_inputs(specs,xi,gradient,previous_update):
    from .batches import expected_active_mask
    require(xi.shape == gradient.shape == previous_update.shape == (len(specs),18),"coefficient inputs must be [B,18]")
    active = expected_active_mask(specs,device=xi.device)
    scales = torch.ones_like(xi)
    modes = torch.zeros(len(specs),18,2,device=xi.device,dtype=torch.int64)
    blocks = torch.zeros_like(xi,dtype=torch.int64)
    modal = torch.tensor([(m,n) for m in (1,2,3) for n in (1,2,3)],device=xi.device,dtype=torch.int64)
    for i,spec in enumerate(specs):
        require(tuple(b.id for b in spec.parameter_blocks) == (("a","b") if spec.name=="wave" else ("a",)),"coefficient tokens support canonical V1 modal block identities")
        for j,block in enumerate(spec.parameter_blocks):
            scales[i,j*9:(j+1)*9] = torch.tensor(block.prior.scale,device=xi.device,dtype=xi.dtype)
            modes[i,j*9:(j+1)*9] = modal
            blocks[i,j*9:(j+1)*9] = j
    return CoefficientTokenBatch(xi,active,modes,blocks,scales,gradient,previous_update)


class CoefficientEncoder(nn.Module):
    """B-compatible coefficient representation; no spatial memory allocation."""
    def __init__(self):
        super().__init__()
        self.mode = nn.Embedding(9,16)
        self.block = nn.Embedding(2,16)
        self.network = nn.Sequential(nn.Linear(9+32,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH),nn.LayerNorm(WIDTH))

    def forward(self,batch):
        module_input(self,batch.xi,"coefficient values")
        active = batch.active
        require(active.shape == batch.xi.shape and active.dtype == torch.bool,"coefficient active-mask mismatch")
        for value in (batch.xi,batch.gradient,batch.previous_update,batch.prior_scale):
            require(value.shape == active.shape and value.device == batch.xi.device and value.dtype == batch.xi.dtype,
                    "coefficient numerical input shape/device/dtype mismatch")
            require(bool(torch.isfinite(value[active]).all()), "nonfinite active coefficient features")
        require(bool((batch.prior_scale[active]>0).all()),"positive coefficient prior scales required")
        require(batch.modes.shape == (*active.shape,2) and batch.modes.dtype == batch.block.dtype == torch.int64,
                "coefficient mode/block identities must be integer tensors")
        require(batch.block.shape == active.shape and batch.block.device == batch.modes.device == batch.xi.device,
                "coefficient identity shape/device mismatch")
        require(bool(((batch.modes[active]>=1)&(batch.modes[active]<=3)).all()),"invalid sine mode identity")
        require(bool(((batch.block[active]>=0)&(batch.block[active]<2)).all()),"invalid coefficient block")
        xi,g,previous = (masked_values(x,active) for x in (batch.xi,batch.gradient,batch.previous_update))
        scale = torch.where(active,batch.prior_scale,torch.ones_like(batch.prior_scale))
        modes = torch.where(active[...,None],batch.modes,torch.ones_like(batch.modes))
        block = masked_values(batch.block,active)
        mode_index = (modes[...,0]-1)*3+(modes[...,1]-1)
        features = torch.stack((xi,g,previous,xi/scale,g*scale,previous/scale,scale.log(),
                                modes[...,0].to(xi.dtype)/3,modes[...,1].to(xi.dtype)/3),dim=-1)
        tokens = self.network(torch.cat((features,self.mode(mode_index),self.block(block)),dim=-1))
        tokens = masked_values(tokens,active)
        return tokens,masked_pool(tokens,active)

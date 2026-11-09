"""Representation encoders only; no field predictor, solver or training loop."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import torch
from torch import nn

from .batches import masked_values
from .core import Status, digest, require
from .graph import Op
from .geometry import boundary_features

WIDTH = 96
OPS = tuple(op.value for op in Op)
STATUSES = (None, *tuple(Status))
ROLES = (None, "state", "source", "initial", "coefficient")
NUMERIC_NAMES = ("rank", "dimension_0", "dimension_1", "unit_L", "unit_T", "unit_U",
                 "spatial_order", "physical_time_order", "literal_0", "literal_1", "literal_2", "literal_3",
                 "literal_count", "known_value", "signed_log_known_value", "known_value_present",
                 "physical_time_present", "equation_root", "boundary_links", "known_zero_boundary_links",
                 "initial_links", "unknown_initial_links", "initial_prior_scale_mean", "initial_components",
                 "input_components", "input_prior_mean", "input_prior_scale_mean", "input_known_fraction")


def module_input(module, tensor, label="input"):
    parameter = next(module.parameters())
    require(tensor.dtype == torch.float32 and parameter.dtype == torch.float32,
            f"{label} and encoder must use explicit float32")
    require(tensor.device == parameter.device, f"{label} device mismatch; transfer explicitly")


def masked_pool(tokens, valid):
    clean = masked_values(tokens, valid)
    return clean.sum(1) / valid.sum(1, keepdim=True).clamp_min(1).to(tokens.dtype)


@dataclass(frozen=True)
class EquationVocabulary:
    quantities: tuple[str, ...]

    def __post_init__(self):
        require(self.quantities == tuple(sorted(set(self.quantities))), "quantity vocabulary must be unique and sorted")

    @classmethod
    def from_specs(cls, specs):
        return cls(tuple(sorted({b.quantity_id for s in specs for b in s.bindings})))

    def index(self, identity):
        if identity is None:
            return 0
        require(identity in self.quantities, f"undeclared encoder quantity identity {identity!r}")
        return 1+self.quantities.index(identity)

    @property
    def identity(self):
        return digest({"version":"typed-representation-v1", "ops":OPS, "quantities":self.quantities,
                       "numeric_attributes":NUMERIC_NAMES, "edge_roles":"operand_index x direction x destination_op"})


@dataclass(frozen=True)
class GraphBatch:
    node_type: torch.Tensor  # [B,V], long
    quantity: torch.Tensor  # explicit vocabulary index, never hashed node ID
    status: torch.Tensor
    role: torch.Tensor
    numeric: torch.Tensor  # [B,V,28]
    valid: torch.Tensor  # [B,V]
    edges: torch.Tensor  # [B,E,2], ordered source/destination indices
    operand_index: torch.Tensor  # [B,E], 0 or 1
    edge_valid: torch.Tensor
    vocabulary_id: str

    def to(self, device):
        return GraphBatch(**{f.name: (getattr(self,f.name).to(device=device) if isinstance(getattr(self,f.name),torch.Tensor)
                                    else getattr(self,f.name)) for f in fields(self)})

    def validate(self, vocabulary):
        b,v = self.valid.shape
        require(self.valid.dtype == self.edge_valid.dtype == torch.bool, "graph masks must be bool")
        require(v > 0 and b > 0 and self.vocabulary_id == vocabulary.identity, "graph vocabulary/batch mismatch")
        require(self.numeric.shape == (b,v,len(NUMERIC_NAMES)), "graph numerical feature shape mismatch")
        require(self.numeric.dtype == torch.float32, "graph attributes require explicit float32")
        for name,maximum in (("node_type",len(OPS)),("quantity",len(vocabulary.quantities)+1),
                             ("status",len(STATUSES)),("role",len(ROLES))):
            value = getattr(self,name)
            require(value.shape == (b,v) and value.dtype == torch.int64, f"invalid graph {name}")
            require(bool(((value[self.valid]>=0)&(value[self.valid]<maximum)).all()), f"unknown graph {name}")
        e = self.edge_valid.shape[1]
        require(self.edges.shape == (b,e,2) and self.operand_index.shape == (b,e), "graph edge shape mismatch")
        require(self.edges.dtype == self.operand_index.dtype == torch.int64, "graph edge indices must be int64")
        require(bool(torch.isfinite(self.numeric[self.valid]).all()), "nonfinite valid graph attributes")
        require(bool(((self.operand_index[self.edge_valid]>=0)&(self.operand_index[self.edge_valid]<2)).all()), "unsupported operand role")
        for row in range(b):
            indices = self.edges[row,self.edge_valid[row]]
            require(bool(((indices>=0)&(indices<v)).all()), "graph edge endpoint outside storage")
            require(bool(self.valid[row,indices].all()), "edge references padded node")
        require(all(getattr(self,f.name).device == self.numeric.device for f in fields(self)
                    if isinstance(getattr(self,f.name),torch.Tensor)), "graph device mismatch")
        return self


def pack_graphs(specs, vocabulary, *, pad_nodes=0, pad_edges=0):
    require(bool(specs), "nonempty graph specification batch required")
    require(type(pad_nodes) is int and type(pad_edges) is int and min(pad_nodes,pad_edges)>=0, "invalid graph padding")
    for spec in specs:
        spec.validate()
    b = len(specs)
    v = max(len(s.graph.nodes) for s in specs)+pad_nodes
    e = max(len(s.graph.edges()) for s in specs)+pad_edges
    ids = {name:torch.zeros(b,v,dtype=torch.int64) for name in ("node_type","quantity","status","role")}
    numeric = torch.zeros(b,v,len(NUMERIC_NAMES),dtype=torch.float32)
    valid,edge_valid = torch.zeros(b,v,dtype=torch.bool),torch.zeros(b,e,dtype=torch.bool)
    edges,operands = torch.zeros(b,e,2,dtype=torch.int64),torch.zeros(b,e,dtype=torch.int64)
    for row,spec in enumerate(specs):
        bindings = {binding.id:binding for binding in spec.bindings}
        blocks = {block.id:block for block in spec.parameter_blocks}
        node_indices = {node.id:i for i,node in enumerate(spec.graph.nodes)}
        for i,node in enumerate(spec.graph.nodes):
            binding = bindings.get(node.binding)
            known = binding.value if binding is not None and binding.value is not None else 0.
            known_present = float(binding is not None and binding.value is not None)
            boundary = [bc for bc in spec.boundaries if binding is not None and bc.target == binding.id]
            initial = [ic for ic in spec.initials if binding is not None and ic.target == binding.id]
            initial_blocks = [blocks[ic.block_id] for ic in initial]
            own_blocks = [block for block in spec.parameter_blocks if binding is not None and block.quantity_binding == binding.id]
            prior = [scale for block in initial_blocks if block.prior for scale in block.prior.scale]
            own_prior = [scale for block in own_blocks if block.prior for scale in block.prior.scale]
            own_means = [mean for block in own_blocks if block.prior for mean in block.prior.mean]
            own_known = [x for block in own_blocks for x in block.known_values]
            values = list(node.values)
            require(len(values)<=4 and node.type.rank<=2, "encoder supports up to four constant entries and rank two")
            feature = [node.type.rank,*(list(node.type.shape)+[0,0])[:2],*node.type.units,
                       node.derivative[1] if node.derivative and node.derivative[0]=="space" else 0,
                       node.derivative[1] if node.derivative and node.derivative[0]=="physical_time" else 0,
                       *(values+[0.]*4)[:4],len(values),known,math.copysign(math.log1p(abs(known)),known),known_present,
                       float(spec.physical_time is not None),float(node.id in spec.graph.roots),len(boundary),
                       sum(bc.status==Status.KNOWN and bc.value==0 for bc in boundary),len(initial),
                       sum(block.status in {Status.UNKNOWN,Status.PARTIAL} for block in initial_blocks),
                       sum(prior)/max(len(prior),1),sum(len(block.component_ids) for block in initial_blocks),
                       len(own_known),sum(own_means)/max(len(own_means),1),sum(own_prior)/max(len(own_prior),1),
                       sum(x is not None for x in own_known)/max(len(own_known),1)]
            require(len(feature)==len(NUMERIC_NAMES), "internal graph feature vocabulary mismatch")
            numeric[row,i] = torch.tensor(feature,dtype=torch.float32)
            ids["node_type"][row,i] = OPS.index(node.op.value)
            ids["quantity"][row,i] = vocabulary.index(node.quantity_id)
            ids["status"][row,i] = STATUSES.index(node.status)
            ids["role"][row,i] = ROLES.index(binding.role if binding else None)
            valid[row,i] = True
        for i,(source,target,operand) in enumerate(spec.graph.edges()):
            edges[row,i] = torch.tensor((node_indices[source],node_indices[target]))
            operands[row,i] = operand
            edge_valid[row,i] = True
    return GraphBatch(**ids,numeric=numeric,valid=valid,edges=edges,operand_index=operands,
                      edge_valid=edge_valid,vocabulary_id=vocabulary.identity).validate(vocabulary)


class TypedMessageLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.relation = nn.Embedding(len(OPS)*2*2,WIDTH)
        self.message = nn.Sequential(nn.Linear(3*WIDTH,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH))
        self.update = nn.Sequential(nn.Linear(2*WIDTH,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH))
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self,h,batch):
        b,v,_ = h.shape
        edges = masked_values(batch.edges,batch.edge_valid)
        operand = masked_values(batch.operand_index,batch.edge_valid)
        source,target = edges[...,0],edges[...,1]
        aggregate,degree = torch.zeros_like(h),torch.zeros(b,v,1,dtype=h.dtype,device=h.device)
        def gather(index):
            return h.gather(1,index[...,None].expand(-1,-1,WIDTH))
        for direction,(src,dst) in enumerate(((source,target),(target,source))):
            destination_type = masked_values(batch.node_type,batch.valid).gather(1,dst)
            relation_id = (destination_type*2+operand)*2+direction
            message = self.message(torch.cat((gather(src),gather(dst),self.relation(relation_id)),dim=-1))
            message = masked_values(message,batch.edge_valid)
            aggregate.scatter_add_(1,dst[...,None].expand(-1,-1,WIDTH),message)
            degree.scatter_add_(1,dst[...,None],batch.edge_valid[...,None].to(h.dtype))
        updated = self.norm(h+self.update(torch.cat((h,aggregate/degree.clamp_min(1)),dim=-1)))
        return masked_values(updated,batch.valid)


@dataclass(frozen=True)
class EquationEncoding:
    nodes: torch.Tensor
    context: torch.Tensor
    valid: torch.Tensor


class EquationEncoder(nn.Module):
    def __init__(self,vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.type_embedding = nn.Embedding(len(OPS),WIDTH)
        self.quantity_embedding = nn.Embedding(len(vocabulary.quantities)+1,WIDTH)
        self.status_embedding = nn.Embedding(len(STATUSES),WIDTH)
        self.role_embedding = nn.Embedding(len(ROLES),WIDTH)
        self.numeric = nn.Sequential(nn.Linear(len(NUMERIC_NAMES),WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH))
        self.norm = nn.LayerNorm(WIDTH)
        self.layers = nn.ModuleList([TypedMessageLayer(),TypedMessageLayer()])

    def forward(self,batch):
        batch.validate(self.vocabulary)
        module_input(self,batch.numeric,"graph numeric attributes")
        clean = lambda value: masked_values(value,batch.valid)
        h = self.norm(self.type_embedding(clean(batch.node_type))+self.quantity_embedding(clean(batch.quantity))+
                      self.status_embedding(clean(batch.status))+self.role_embedding(clean(batch.role))+self.numeric(clean(batch.numeric)))
        h = clean(h)
        for layer in self.layers:
            h = layer(h,batch)
        return EquationEncoding(h,masked_pool(h,batch.valid),batch.valid)


class FourierFeatures(nn.Module):
    """Low/high frequencies 1,2,4,8,16,32; raw coordinates retained separately."""
    def __init__(self,frequencies=(1.,2.,4.,8.,16.,32.)):
        super().__init__()
        require(len(frequencies)>0 and all(math.isfinite(f) and f>0 for f in frequencies),"positive Fourier frequencies required")
        self.register_buffer("frequencies",torch.tensor(frequencies,dtype=torch.float32))

    def forward(self,x):
        phase = 2*math.pi*x[...,None]*self.frequencies
        return torch.cat((phase.sin(),phase.cos()),dim=-1).flatten(-2)


def boundary_distances(position):
    # Compatibility API: geometry owns the actual boundary feature definition.
    return boundary_features(position)


class CoordinateFeatures(nn.Module):
    size = 44  # position 2 + boundary 4 + spatial Fourier 24 + time 1 + Fourier 12 + presence 1

    def __init__(self,geometry_provider=None,*,frequencies=(1.,2.,4.,8.,16.,32.)):
        super().__init__()
        self.geometry_provider = geometry_provider
        require(geometry_provider is None or geometry_provider.boundary_feature_size==4,
                "V1 coordinate feature contract requires four geometry boundary features")
        self.fourier = FourierFeatures(frequencies)
        self.size = 8+6*len(frequencies)

    def forward(self,position,physical_time,time_present,valid):
        p = masked_values(position,valid)
        presence = time_present & valid
        time = masked_values(physical_time,presence)
        # Mask AFTER Fourier encoding too: cos(0)=1 must not fabricate physical time.
        time_fourier = masked_values(self.fourier(time),presence)
        x = torch.cat((p,boundary_features(p,self.geometry_provider),self.fourier(p),time,time_fourier,presence[...,None].to(p.dtype)),dim=-1)
        return masked_values(x,valid)


@dataclass(frozen=True)
class ObservationEncoding:
    tokens: torch.Tensor
    summary: torch.Tensor
    position: torch.Tensor
    physical_time: torch.Tensor
    time_present: torch.Tensor
    quantity_id: torch.Tensor
    log_noise: torch.Tensor
    valid: torch.Tensor


class ObservationEncoder(nn.Module):
    def __init__(self,normalizers,*,coordinate=None):
        super().__init__()
        self.normalizer_identity = normalizers.identity
        self.families = ("poisson","heat","wave")
        scales = [[normalizers.scale(p,0),normalizers.scale(p,1) if p=="wave" else 1.] for p in self.families]
        self.register_buffer("quantity_scales",torch.tensor(scales,dtype=torch.float32))
        self.coordinate = CoordinateFeatures() if coordinate is None else coordinate
        self.quantity = nn.Embedding(2,16)
        # normalized value, normalized sigma, log normalized sigma; no truth metadata.
        self.network = nn.Sequential(nn.Linear(self.coordinate.size+16+3,WIDTH),nn.SiLU(),
                                     nn.Linear(WIDTH,WIDTH),nn.LayerNorm(WIDTH))

    def forward(self,batch,specs):
        batch.validate(specs)
        module_input(self,batch.position,"observations")
        clean = batch.masked()
        require(all(s.name in self.families for s in specs), "unsupported observation normalization family")
        family = torch.tensor([self.families.index(s.name) for s in specs],device=clean.position.device)
        scales = self.quantity_scales[family[:,None],clean.quantity_id]
        normalized_value = clean.value/scales[...,None]
        normalized_sigma = clean.sigma/scales[...,None]
        safe_sigma = torch.where(clean.valid[...,None],normalized_sigma,torch.ones_like(normalized_sigma))
        log_noise = safe_sigma.log()
        features = torch.cat((self.coordinate(clean.position,clean.physical_time,clean.time_present,clean.valid),
                              self.quantity(clean.quantity_id),normalized_value,normalized_sigma,log_noise),dim=-1)
        tokens = masked_values(self.network(features),clean.valid)
        return ObservationEncoding(tokens,masked_pool(tokens,clean.valid),clean.position,clean.physical_time,
                                   clean.time_present,clean.quantity_id,masked_values(log_noise,clean.valid),clean.valid)


class ConditionalQueryEncoder(nn.Module):
    """Coordinate features with fixed global conditioning; returns features, no scalar field head."""
    def __init__(self):
        super().__init__()
        self.coordinate = CoordinateFeatures()
        self.quantity = nn.Embedding(2,16)
        self.project = nn.Sequential(nn.Linear(self.coordinate.size+16,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH))
        self.condition = nn.Linear(2*WIDTH,2*WIDTH)
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self,query,specs,equation_context,observation_context):
        query.validate(specs)
        module_input(self,query.position,"queries")
        q = query.masked()
        require(equation_context.shape == observation_context.shape == (len(specs),WIDTH), "query context shape mismatch")
        module_input(self,equation_context,"query equation context")
        module_input(self,observation_context,"query observation context")
        features = self.project(torch.cat((self.coordinate(q.position,q.physical_time,q.time_present,q.valid),
                                           self.quantity(q.quantity_id)),dim=-1))
        gamma,beta = self.condition(torch.cat((equation_context,observation_context),dim=-1)).chunk(2,dim=-1)
        return masked_values(self.norm(features*(1+.1*gamma[:,None].tanh())+beta[:,None]),q.valid)

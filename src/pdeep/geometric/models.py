"""Independent geometry inverse models. Predictions never invoke a PDE solver."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from ..core import require,digest
from .representation import (TypedGraph,MeshEncoder,Observations,Attention,Residual,
                             Coordinates,layout,tensor,interpolate,retrieval_weights)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str='A_GEO'
    capacity: str='medium'
    width: int=192
    heads: int=6
    graph_layers: int=2
    geometry_layers: int=2
    observation_layers: int=2
    memory_blocks: int=2
    decoder_layers: int=4
    coarse: int=64
    fine: int=256
    spatial: bool=True
    geometry: bool=True
    physics: bool=True
    multiscale: bool=True
    velocity: str='independent'
    def __post_init__(self):
        require(self.architecture in {'A_GEO','C_GEO'} and self.width%self.heads==0,'invalid geometry model identity/heads')
        require(self.capacity in {'medium','large','verification'} and min(self.graph_layers,self.geometry_layers,self.observation_layers,self.memory_blocks,self.decoder_layers,self.coarse,self.fine)>0,'invalid geometry capacity')
        require(self.velocity in {'independent','time_derivative'},'explicit wave velocity coupling required')
    @classmethod
    def preset(cls,architecture,capacity='medium',**kw):
        if capacity=='large':return cls(architecture,capacity,width=256,heads=8,**kw)
        if capacity=='verification':return cls(architecture,capacity,width=32,heads=4,graph_layers=1,geometry_layers=1,observation_layers=1,memory_blocks=1,decoder_layers=2,coarse=8,fine=16,**kw)
        return cls(architecture,capacity,**kw)
    @property
    def identity(self):return digest(asdict(self))


@dataclass
class Context:
    problem: object
    layout: object
    equation: torch.Tensor
    global_context: torch.Tensor
    geometry_nodes: torch.Tensor
    memory: torch.Tensor
    latent: torch.Tensor
    attention: list


class DirectDecoder(nn.Module):
    def __init__(self,d,layers):
        super().__init__();self.input=nn.Linear(Coordinates.size+5*d,d);self.layers=nn.ModuleList(Residual(d,2) for _ in range(layers));self.norm=nn.LayerNorm(d)
    def forward(self,coordinates,equation,global_context,local,latent):
        h=self.input(torch.cat((coordinates,equation.expand(len(coordinates),-1),global_context.expand(len(coordinates),-1),local,latent.expand(len(coordinates),-1)),-1))
        for layer in self.layers:h=layer(h)
        return self.norm(h)


class FiLMLayer(nn.Module):
    def __init__(self,d):
        super().__init__();self.norm=nn.LayerNorm(d);self.modulation=nn.Linear(d,2*d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,d))
    def forward(self,h,c):
        gamma,beta=(.1*self.modulation(c).tanh()).chunk(2,-1)
        return h+self.ff(self.norm(h)*(1+gamma)+beta)


class FiLMField(nn.Module):
    def __init__(self,d,layers):
        super().__init__();self.input=nn.Linear(Coordinates.size,d);self.condition=nn.Linear(5*d,d);self.layers=nn.ModuleList(FiLMLayer(d) for _ in range(layers));self.norm=nn.LayerNorm(d)
    def forward(self,coordinates,equation,global_context,local,latent):
        h=self.input(coordinates);c=self.condition(torch.cat((equation.expand(len(h),-1),global_context.expand(len(h),-1),local,latent.expand(len(h),-1)),-1))
        for layer in self.layers:h=layer(h,c)
        return self.norm(h)


class GeometryModel(nn.Module):
    """A: memory + direct residual decoder. C: global/local FiLM field.

    Flexible auxiliary field heads predict f/u0/v0 at arbitrary x; no fixed xi
    slots. Deterministic latent is one vector per case, reused across chunks.
    An optional externally supplied latent can replace it in a future sampler.
    """
    def __init__(self,config,normalizers):
        super().__init__();self.config=config;self.normalizers=normalizers;d=config.width
        self.equation=TypedGraph(d,config.graph_layers);self.geometry=MeshEncoder(d,config.geometry_layers)
        self.observation=Observations(d,config.heads,config.observation_layers)
        self.context=nn.Sequential(nn.Linear(3*d+3,d),nn.SiLU(),nn.Linear(d,d));self.anchor_input=nn.Linear(2*d+3,d)
        self.memory_layers=nn.ModuleList(Attention(d,config.heads) for _ in range(config.memory_blocks))
        self.observation_cross=Attention(d,config.heads);self.geometry_cross=Attention(d,config.heads)
        self.coordinates=Coordinates();self.latent_projection=nn.Linear(d,d)
        self.decoder=(DirectDecoder if config.architecture=='A_GEO' else FiLMField)(d,config.decoder_layers)
        self.u_head=nn.Linear(d,1);self.v_head=nn.Linear(d,1);self.aux_heads=nn.ModuleDict({k:nn.Linear(d,1) for k in ('f','u0','v0')})

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def inventory(self):return {k:sum(p.numel() for p in m.parameters()) for k,m in self.named_children()}

    def prepare_context(self,problems,observations,*,latent=None):
        require(observations.position.dtype==torch.float32,'neural geometry inputs require float32')
        nodes,equations=self.equation(problems,observations.position.device);contexts=[]
        for row,problem in enumerate(problems):
            lay=layout(problem.mesh,self.config.coarse,self.config.fine);eq=equations[row]
            geom,gpool=self.geometry(lay,eq,problem.conditions)
            if not self.config.geometry:geom=geom*0;gpool=gpool*0
            obs,opool,attn=self.observation(lay,problem,observations,row,eq,self.normalizers)
            # Supplied known numerical coefficient, time presence and area enter
            # downstream paths in addition to the typed equation graph.
            known=eq.new_tensor([problem.coefficient,float(problem.physical_time is not None),problem.mesh.cell_areas.sum()])
            global_context=self.context(torch.cat((eq,gpool,opool,known)))
            anchor_geo=geom[torch.as_tensor(lay.anchors,device=eq.device)]
            positions=tensor(problem.mesh.vertices[lay.anchors],eq);scale=eq.new_tensor([0.]*len(lay.coarse)+[1.]*len(lay.fine))[:,None]
            memory=self.anchor_input(torch.cat((anchor_geo,global_context.expand(len(positions),-1),positions,scale),-1))
            # Domain graph bias never connects fictitious vertices across a hole.
            L2=problem.mesh.cell_areas.sum();bias=-tensor(lay.distance2[lay.anchors],eq)/(2*.3**2*L2)
            node_bias=-tensor(lay.distance2.T,eq)/(2*.25**2*L2)+tensor(np.log(lay.mass_weights/lay.mass_weights.sum()),eq)[None,:]
            memory,wg=self.geometry_cross(memory,geom,bias=node_bias);attn.append(wg)
            if len(obs):
                select=observations.valid[row];x=observations.position[row,select]
                d2=interpolate(problem.mesh,tensor(lay.distance2,x),x).T
                memory,wo=self.observation_cross(memory,obs,bias=-d2/(2*.25**2*L2));attn.append(wo)
            for layer in self.memory_layers:memory,w=layer(memory,memory,bias=bias);attn.append(w)
            z=self.latent_projection(global_context) if latent is None else latent[row]
            require(z.shape==(self.config.width,),'one coherent latent vector per case required')
            contexts.append(Context(problem,lay,eq,global_context,geom,memory,z,attn))
        return tuple(contexts)

    def _hidden(self,context,x,t):
        lay=context.layout;c=len(lay.coarse)
        coords=self.coordinates(lay,x,t,context.problem.physical_time is not None)
        if self.config.spatial:
            w=retrieval_weights(lay,x);coarse=w[:,:c]@context.memory[:c];fine=w[:,c:]@context.memory[c:]
            if not self.config.multiscale:coarse=coarse*0
        else:coarse=context.global_context.expand(len(x),-1);fine=torch.zeros_like(coarse)
        local=torch.cat((coarse,fine),-1)
        return self.decoder(coords,context.equation,context.global_context,local,context.latent)

    def evaluate(self,context,x,t=None,quantity='u'):
        require(quantity in {'u','v','f','u0','v0'},'invalid geometry quantity')
        dynamic=context.problem.physical_time is not None
        require((t is not None)==dynamic,'stationary physical time must be absent')
        if t is None:t=x.new_zeros((len(x),1))
        if quantity in {'u0','v0','f'}:
            return self.aux_heads[quantity](self._hidden(context,x,torch.zeros_like(t)))*self.normalizers.scale(context.problem.name,quantity)
        if quantity=='v' and self.config.velocity=='time_derivative':
            with torch.enable_grad():
                if not t.requires_grad:t=t.clone().requires_grad_(True)
                u=self.u_head(self._hidden(context,x,t))*self.normalizers.scale(context.problem.name,'u')
                return torch.autograd.grad(u.sum(),t,create_graph=True)[0]
        h=self._hidden(context,x,t);head=self.u_head if quantity=='u' else self.v_head
        return head(h)*self.normalizers.scale(context.problem.name,quantity)

    def predict(self,contexts,queries,*,chunk=256):
        rows=[]
        for i,context in enumerate(contexts):
            pieces=[]
            for a in range(0,queries.valid.shape[1],chunk):
                end=min(a+chunk,queries.valid.shape[1]);valid=queries.valid[i,a:end];x=queries.position[i,a:end][valid];q=queries.quantity_id[i,a:end][valid]
                t=queries.physical_time[i,a:end][valid] if context.problem.physical_time is not None else None
                vals=x.new_zeros((len(x),1))
                for k,name in ((0,'u'),(1,'v')):
                    choose=q==k
                    if choose.any():vals[choose]=self.evaluate(context,x[choose],None if t is None else t[choose],name)
                padded=queries.position.new_zeros((end-a,1));padded[valid]=vals;pieces.append(padded)
            rows.append(torch.cat(pieces,0) if pieces else queries.position.new_zeros((0,1)))
        return torch.stack(rows)

    def forward(self,problems,observations,queries,*,chunk=256):return self.predict(self.prepare_context(problems,observations),queries,chunk=chunk)


def create(architecture,normalizers,capacity='medium',*,seed=0,device='cpu',**ablations):
    from ..runtime import resolve_device
    resolve_device(device,'float32')
    config=ModelConfig.preset(architecture,capacity,**ablations)
    # Independent instances, reproducible construction, no global CPU RNG drift.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=GeometryModel(config,normalizers)
    return model.to(device)

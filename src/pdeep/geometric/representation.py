"""Topology-aware H1 coordinate path. All geometry operations live here.

Anchor selection/triangle selection are discrete preprocessing. Query barycentric
weights are recomputed in torch, preserving coordinate derivatives. Interpolated
nodal graph-distance squared is continuous across elements, piecewise affine;
Gaussian weights are smooth INSIDE each element, globally H1/C0, not C1 or C2.
Weak physics uses element quadrature and needs first weak derivatives. Strong
second derivatives on mesh interfaces are expressly unsupported.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from scipy.sparse.csgraph import dijkstra
from ..core import require
from ..encoding import EquationVocabulary,pack_graphs,OPS,STATUSES,ROLES,NUMERIC_NAMES,FourierFeatures
from ..benchmarks import benchmark


def tensor(a,like):
    from ..learning_diagnostics.runtime import current
    cache=current()
    return cache.convert(a,like) if cache is not None else torch.as_tensor(np.array(a,copy=True),device=like.device,dtype=like.dtype)

def pool(h,mask):return torch.where(mask[...,None],h,0.).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

def fps(mesh,count):
    boundary=np.unique(mesh.boundary_edges);candidates=np.setdiff1d(np.arange(len(mesh.vertices)),boundary)
    require(len(candidates)>0,'mesh has no interior anchors; refine mesh')
    # Coordinate tie-breaking, independent of vertex indexing.
    order=np.lexsort((mesh.vertices[candidates,1],mesh.vertices[candidates,0]));candidates=candidates[order]
    chosen=[];distance=np.full(len(candidates),np.inf)
    for _ in range(min(count,len(candidates))):
        j=int(np.argmax(distance));node=int(candidates[j]);chosen.append(node)
        distance=np.minimum(distance,dijkstra(mesh.graph,indices=node)[candidates]);distance[np.isin(candidates,chosen)]=-1
    return np.array(chosen)


@dataclass
class Layout:
    mesh: object
    coarse: np.ndarray
    fine: np.ndarray
    distance2: np.ndarray
    node_features: np.ndarray
    mass_weights: np.ndarray
    @property
    def anchors(self):return np.r_[self.coarse,self.fine]
    @property
    def counts(self):return len(self.coarse),len(self.fine)
    @property
    def valid(self):return np.ones(len(self.anchors),dtype=bool)


@lru_cache(maxsize=64)
def _layout(mesh_id,coarse,fine):
    mesh=_MESHES[mesh_id];c=fps(mesh,coarse);f=fps(mesh,fine);anchors=np.r_[c,f]
    dist=dijkstra(mesh.graph,indices=anchors).T
    geo=mesh.features(mesh.vertices);L=np.sqrt(mesh.cell_areas.sum());lo=np.min(mesh.vertices,0);extent=np.ptp(mesh.vertices,axis=0)
    mass=np.zeros(len(mesh.vertices));np.add.at(mass,mesh.cells.ravel(),np.repeat(mesh.cell_areas/3,3))
    features=np.column_stack((mesh.vertices,(mesh.vertices-lo)/extent,geo['distance']/L,geo['normal'],geo['normal_defined'],
                              geo['boundary_tag']/max(mesh.boundary_tags),np.log(np.maximum(geo['local_h']/L,1e-8)),mass/L**2))
    return Layout(mesh,c,f,dist**2,features,mass)


_MESHES={}
def layout(mesh,coarse=64,fine=256):
    _MESHES[mesh.mesh_id]=mesh;return _layout(mesh.mesh_id,coarse,fine)


def barycentric(mesh,x):
    from ..learning_diagnostics.runtime import current
    cache=current()
    if cache is not None:return cache.barycentric(mesh,x)
    # The integer cell index has no derivative; coordinates DO. On a shared
    # edge both incident interpolants have identical trace, unlike nearest-node.
    tolerance=4*torch.finfo(x.dtype).eps*max(1.,np.abs(mesh.vertices).max())
    ids,_=mesh.locate(x.detach().cpu().double().numpy(),coordinate_tolerance=tolerance);tri=torch.as_tensor(mesh.cells[ids],device=x.device,dtype=torch.long)
    verts=tensor(mesh.vertices[mesh.cells[ids]],x);A=torch.stack((verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]),-1)
    uv=torch.linalg.solve(A,(x-verts[:,0])[...,None])[...,0]
    return tri,torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)


def interpolate(mesh,nodal,x):
    tri,w=barycentric(mesh,x);return (nodal[tri]*w[...,None]).sum(1)


def retrieval_weights(lay,x,coarse_length=.30,fine_length=.15):
    d2=interpolate(lay.mesh,tensor(lay.distance2,x),x);c=len(lay.coarse);L2=lay.mesh.cell_areas.sum()
    return torch.cat((torch.softmax(-d2[:,:c]/(2*coarse_length**2*L2),-1),torch.softmax(-d2[:,c:]/(2*fine_length**2*L2),-1)),-1)


class Residual(nn.Module):
    def __init__(self,d,expansion=4):
        super().__init__();self.norm=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,expansion*d),nn.SiLU(),nn.Linear(expansion*d,d))
    def forward(self,h):return h+self.ff(self.norm(h))


class Attention(nn.Module):
    """Native explicit attention supports N=0 and coordinate double backward."""
    def __init__(self,d,heads):
        super().__init__();self.d=d;self.heads=heads;self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d);self.norm=nn.LayerNorm(d);self.ff=Residual(d)
    def forward(self,h,source,*,bias=None,valid=None):
        if source.shape[-2]==0:return self.ff(h),h.new_zeros((self.heads,h.shape[-2],0))
        d=self.d//self.heads
        split=lambda x:x.reshape(x.shape[0],self.heads,d).transpose(0,1)
        q,k,v=split(self.q(self.norm(h))),split(self.k(source)),split(self.v(source));logits=q@k.transpose(-1,-2)/np.sqrt(d)
        if bias is not None:logits=logits+bias
        if valid is not None:logits=logits.masked_fill(~valid[None,None,:],-torch.inf)
        weights=logits.softmax(-1);h=h+self.out((weights@v).transpose(0,1).reshape(-1,self.d))
        return self.ff(h),weights


class TypedGraph(nn.Module):
    def __init__(self,d,layers):
        super().__init__();self.vocabulary=EquationVocabulary.from_specs(tuple(benchmark(p) for p in ('poisson','heat','wave')))
        self.types=nn.Embedding(len(OPS),d);self.quantities=nn.Embedding(len(self.vocabulary.quantities)+1,d);self.status=nn.Embedding(len(STATUSES),d);self.roles=nn.Embedding(len(ROLES),d)
        self.numeric=nn.Sequential(nn.Linear(len(NUMERIC_NAMES),d),nn.SiLU(),nn.Linear(d,d));self.relations=nn.Embedding(len(OPS)*4,d)
        self.messages=nn.ModuleList(nn.Sequential(nn.Linear(3*d,2*d),nn.SiLU(),nn.Linear(2*d,d)) for _ in range(layers));self.updates=nn.ModuleList(Residual(d) for _ in range(layers))
    def forward(self,problems,device):
        from ..learning_diagnostics.runtime import current
        cache=current()
        b=cache.graphs(problems,self.vocabulary,device) if cache is not None else pack_graphs(tuple(p.symbolic for p in problems),self.vocabulary).to(device)
        h=self.types(b.node_type)+self.quantities(b.quantity)+self.status(b.status)+self.roles(b.role)+self.numeric(b.numeric)
        h=h*b.valid[...,None]
        for message,update in zip(self.messages,self.updates):
            aggregate=torch.zeros_like(h);degree=torch.zeros_like(h[...,:1])
            for direction in (0,1):
                src=b.edges[...,direction];dst=b.edges[...,1-direction];gather=lambda i:h.gather(1,i[...,None].expand(-1,-1,h.shape[-1]))
                kind=b.node_type.gather(1,dst);rel=self.relations((kind*2+b.operand_index)*2+direction)
                msg=message(torch.cat((gather(src),gather(dst),rel),-1))*b.edge_valid[...,None]
                aggregate.scatter_add_(1,dst[...,None].expand_as(msg),msg);degree.scatter_add_(1,dst[...,None],b.edge_valid[...,None].to(h.dtype))
            h=update(h+aggregate/degree.clamp_min(1))*b.valid[...,None]
        return h,pool(h,b.valid)


class MeshEncoder(nn.Module):
    def __init__(self,d,layers):
        super().__init__();self.input=nn.Linear(14,d);self.messages=nn.ModuleList(nn.Sequential(nn.Linear(3*d+3,2*d),nn.SiLU(),nn.Linear(2*d,d)) for _ in range(layers));self.updates=nn.ModuleList(Residual(d) for _ in range(layers))
    def forward(self,lay,context,conditions):
        from ..learning_diagnostics.runtime import current
        cache=current()
        if cache is not None:
            features,src,dst,ef,deg,mass=cache.mesh_inputs(lay,context,conditions)
            h=self.input(features)+context
        else:
            tags=lay.mesh.features(lay.mesh.vertices)['boundary_tag']
            bc=np.array([[float(conditions[int(k)].kind=='DIRICHLET'),float(conditions[int(k)].kind=='NEUMANN'),conditions[int(k)].value] for k in tags])
            h=self.input(tensor(np.column_stack((lay.node_features,bc)),context))+context;edge=np.concatenate((lay.mesh.edge_index,lay.mesh.edge_index[::-1]),1);src,dst=torch.as_tensor(edge,device=h.device,dtype=torch.long)
            delta=tensor(lay.mesh.vertices[edge[0]]-lay.mesh.vertices[edge[1]],h)/np.sqrt(lay.mesh.cell_areas.sum());ef=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
            deg=torch.bincount(dst,minlength=len(h)).clamp_min(1).to(h.dtype)
        for message,update in zip(self.messages,self.updates):
            msg=message(torch.cat((h[src],h[dst],ef,context.expand(len(src),-1)),-1));agg=torch.zeros_like(h).index_add(0,dst,msg)/deg[:,None];h=update(h+agg)
        mass=mass if cache is not None else tensor(lay.mass_weights,h);return h,(h*mass[:,None]).sum(0)/mass.sum()


class Coordinates(nn.Module):
    size=45 # raw xy(2), normalized xy(2), Fourierxy(24), time(1)+Fourier(12)+presence(1), distance/normal(3)
    def __init__(self):super().__init__();self.fourier=FourierFeatures()
    def forward(self,lay,x,t,present):
        lo=tensor(lay.mesh.vertices.min(0),x);span=tensor(np.ptp(lay.mesh.vertices,axis=0),x);norm=(x-lo)/span
        tf=self.fourier(t) if present else torch.zeros((len(x),12),device=x.device,dtype=x.dtype)
        time=t if present else torch.zeros_like(t)
        # Nodal distance/normal interpolation is continuous and H1. Normals here
        # are only geometry descriptors, not imposed boundary conditions.
        local=interpolate(lay.mesh,tensor(lay.node_features[:,[4,5,6]],x),x)
        return torch.cat((x,norm,self.fourier(norm),time,tf,torch.full_like(time,float(present)),local),-1)


class Observations(nn.Module):
    def __init__(self,d,heads,layers):
        super().__init__();self.coordinate=Coordinates();self.quantity=nn.Embedding(2,d);self.input=nn.Linear(Coordinates.size+3,d);self.layers=nn.ModuleList(Attention(d,heads) for _ in range(layers))
    def forward(self,lay,problem,obs,row,context,normalizers):
        select=obs.valid[row];x=obs.position[row,select];t=obs.physical_time[row,select];q=obs.quantity_id[row,select]
        if not len(x):return context.new_zeros((0,len(context))),torch.zeros_like(context),[]
        require(bool(torch.isfinite(obs.value[row,select]).all()) and bool(torch.isfinite(obs.sigma[row,select]).all()) and bool((obs.sigma[row,select]>0).all()),'invalid observed value/noise')
        require(bool((obs.time_present[row,select]==(problem.physical_time is not None)).all()),'invalid physical-time presence')
        scales=context.new_tensor([normalizers.scale(problem.name,0),normalizers.scale(problem.name,1) if problem.name=='wave' else 1.])[q,None]
        noise=obs.sigma[row,select]/scales;values=obs.value[row,select]/scales
        coords=self.coordinate(lay,x,t,problem.physical_time is not None)
        h=self.input(torch.cat((coords,values,noise,noise.log()),-1))+self.quantity(q)+context;weights=[]
        for layer in self.layers:h,w=layer(h,h);weights.append(w)
        return h,h.mean(0),weights

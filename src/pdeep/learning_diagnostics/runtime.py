"""Opt-in immutable geometry/graph INPUT caches. Never cache encoded activations.

ContextVar scopes prevent interference with legacy or parallel instances. Keys
include geometry/BC/coefficient, device and dtype. No hidden targets are registered.
"""
from contextvars import ContextVar
from contextlib import contextmanager
import hashlib
import numpy as np
import torch

_ACTIVE=ContextVar('geometry_input_runtime',default=None)
def current():return _ACTIVE.get()
@contextmanager
def using(runtime):
    token=_ACTIVE.set(runtime)
    try:yield runtime
    finally:_ACTIVE.reset(token)

class InputCache:
    def __init__(self):self.arrays={};self.allowed=set();self.meshes={};self.graph_batches={};self.hits=0;self.misses=0;self.query_geometry={}
    @staticmethod
    def key(a):
        a=np.asarray(a);return a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).digest()
    def register(self,lay):
        for a in (lay.distance2,lay.node_features,lay.mass_weights,lay.mesh.vertices[lay.anchors],lay.mesh.vertices.min(0),np.ptp(lay.mesh.vertices,axis=0),lay.node_features[:,[4,5,6]]):self.allowed.add(self.key(a))
    def convert(self,a,like):
        key=self.key(a)
        if key not in self.allowed:return torch.as_tensor(np.array(a,copy=True),device=like.device,dtype=like.dtype)
        key=(key,str(like.device),str(like.dtype))
        if key not in self.arrays:self.arrays[key]=torch.as_tensor(np.array(a,copy=True),device=like.device,dtype=like.dtype);self.misses+=1
        else:self.hits+=1
        return self.arrays[key]
    def mesh_inputs(self,lay,like,conditions):
        self.register(lay);key=(lay.mesh.mesh_id,str(like.device),str(like.dtype),tuple((k,v.kind,v.value) for k,v in sorted(conditions.items())))
        if key in self.meshes:self.hits+=1;return self.meshes[key]
        tags=lay.mesh.features(lay.mesh.vertices)['boundary_tag']
        bc=np.array([[float(conditions[int(k)].kind=='DIRICHLET'),float(conditions[int(k)].kind=='NEUMANN'),conditions[int(k)].value] for k in tags])
        t=lambda a:torch.as_tensor(np.array(a,copy=True),device=like.device,dtype=like.dtype)
        features=t(np.column_stack((lay.node_features,bc)));edge=np.concatenate((lay.mesh.edge_index,lay.mesh.edge_index[::-1]),1)
        src,dst=torch.as_tensor(edge,device=like.device,dtype=torch.long)
        delta=t(lay.mesh.vertices[edge[0]]-lay.mesh.vertices[edge[1]])/np.sqrt(lay.mesh.cell_areas.sum());ef=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
        deg=torch.bincount(dst,minlength=len(features)).clamp_min(1).to(like.dtype);mass=t(lay.mass_weights)
        self.meshes[key]=(features,src,dst,ef,deg,mass);self.misses+=1;return self.meshes[key]
    def graphs(self,problems,vocabulary,device):
        from ..encoding import pack_graphs
        key=(tuple(p.identity for p in problems),vocabulary.identity,str(device))
        if key not in self.graph_batches:self.graph_batches[key]=pack_graphs(tuple(p.symbolic for p in problems),vocabulary).to(device);self.misses+=1
        else:self.hits+=1
        return self.graph_batches[key]

    def clear_queries(self):self.query_geometry.clear()
    def barycentric(self,mesh,x):
        # Hold the input itself until this update ends, so CUDA allocator pointer
        # reuse cannot turn a new point set into a stale-cache hit.
        key=(mesh.mesh_id,id(x),x._version,str(x.device),str(x.dtype))
        if key not in self.query_geometry:
            tolerance=4*torch.finfo(x.dtype).eps*max(1.,np.abs(mesh.vertices).max())
            ids,_=mesh.locate(x.detach().cpu().double().numpy(),coordinate_tolerance=tolerance)
            tri=torch.as_tensor(mesh.cells[ids],device=x.device,dtype=torch.long)
            verts=torch.as_tensor(np.array(mesh.vertices[mesh.cells[ids]],copy=True),device=x.device,dtype=x.dtype)
            A=torch.stack((verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]),-1)
            self.query_geometry[key]=(x,tri,verts,A);self.misses+=1
        else:self.hits+=1
        _,tri,verts,A=self.query_geometry[key]
        # Recompute differentiable weights every call; cache no learned activation.
        uv=torch.linalg.solve(A,(x-verts[:,0])[...,None])[...,0]
        return tri,torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)

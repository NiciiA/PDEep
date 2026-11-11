"""Mapped tensor Gauss/Duffy triangular quadrature and tagged edge Gauss rules."""
from dataclasses import dataclass
import numpy as np
from ..core import require

@dataclass(frozen=True)
class Quadrature:
    position: np.ndarray
    weight: np.ndarray
    cell_id: np.ndarray
    barycentric: np.ndarray
    region_tag: np.ndarray
    normal: np.ndarray | None=None
    boundary_tag: np.ndarray | None=None


def cells(mesh,order=4):
    require(order>=2,'at least second-order Gauss/Duffy rule required')
    z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    r,s=np.meshgrid(z,z,indexing='ij');wr,ws=np.meshgrid(w,w,indexing='ij')
    bary=np.column_stack((1-r.ravel()-(1-r.ravel())*s.ravel(),r.ravel(),(1-r.ravel())*s.ravel()))
    reference=(wr*ws*(1-r)).ravel();n=len(bary)
    p=np.einsum('ki,tij->tkj',bary,mesh.vertices[mesh.cells])
    return Quadrature(p.reshape(-1,2),(2*mesh.cell_areas[:,None]*reference).ravel(),np.repeat(np.arange(len(mesh.cells)),n),
                      np.tile(bary,(len(mesh.cells),1)),np.repeat(mesh.region_tags,n))


def boundary(mesh,order=4,tags=None):
    z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    ids=np.arange(len(mesh.boundary_edges));ids=ids if tags is None else ids[np.isin(mesh.boundary_tags,list(tags))]
    e=mesh.vertices[mesh.boundary_edges[ids]];p=e[:,0,None]*(1-z)[None,:,None]+e[:,1,None]*z[None,:,None]
    return Quadrature(p.reshape(-1,2),(mesh.boundary_lengths[ids,None]*w).ravel(),np.repeat(ids,order),
                      np.tile(np.column_stack((1-z,z)),(len(ids),1)),np.zeros(len(ids)*order,dtype=int),
                      np.repeat(mesh.boundary_normals[ids],order,axis=0),np.repeat(mesh.boundary_tags[ids],order))

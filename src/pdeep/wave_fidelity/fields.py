"""Versioned smooth-field candidate. NEVER replaces the canonical G1 P1 prior.

Compact polynomial bumps are globally C3, exactly zero near the boundary, with
known covariance. Their support radius is an explicit physical bandwidth control.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from ..core import require,digest

VERSION='G1-wave-C3-compact-Gaussian-candidate-v2'


@lru_cache(maxsize=32)
def _canonical_geometry(shape):
    from ..geometric.domain import mesh_shape
    return mesh_shape(shape,.08)


@dataclass(frozen=True)
class SmoothField:
    centers: np.ndarray
    radius: float
    coefficients: np.ndarray
    geometry_id: str
    amplitude: float=1.
    def __call__(self,x,t=0.):
        r2=np.sum((np.asarray(x)[:,None]-self.centers[None])**2,-1)/self.radius**2
        return self.amplitude*(np.maximum(0.,1-r2)**4)@self.coefficients/np.sqrt(len(self.centers))
    @property
    def identity(self):return digest((VERSION,self.geometry_id,self.centers.tolist(),self.radius,self.coefficients.tolist(),self.amplitude))
    @property
    def specification(self):return {'version':VERSION,'geometry_id':self.geometry_id,'centers':self.centers.tolist(),'support_radius':self.radius,
         'coefficients':'independent standard normal; deterministic keyed realization','amplitude':self.amplitude,'regularity':'C3, compactly supported inside polygon',
         'covariance':'amplitude^2/K * sum_j bump_j(x)*bump_j(y)','field_identity':self.identity}


def centers(mesh,*,relative_radius=.18,count=32):
    require(.1<=relative_radius<=.3 and count>=4,'declared smooth candidate bandwidth/count')
    # Physical centers are defined on one canonical mesh, never on a requested
    # propagation discretization. Calling pair() after remeshing must not draw
    # a different continuous field.
    mesh=_canonical_geometry(mesh.shape)
    from ..geometric.portable import stored_basis
    frozen=stored_basis(mesh,relative_radius,count)
    if frozen is not None:return frozen
    radius=relative_radius*np.sqrt(mesh.cell_areas.sum());seed=int(digest((VERSION,mesh.geometry_id,'centers',relative_radius,count))[:16],16)
    # Same represented polygon, deterministic physical candidates: sampling on
    # ONE canonical mesh only; caller freezes centers before mesh refinement.
    stream=np.random.default_rng(seed);x=mesh.sample(8192,stream);distance=mesh.features(x)['distance'];x=x[distance>radius*(1+1e-10)]
    require(len(x)>=count,'geometry has insufficient interior clearance for candidate radius; do not silently shrink supports')
    selected=[];d=np.full(len(x),np.inf)
    for _ in range(count):
        i=int(np.argmax(d));selected.append(i);d=np.minimum(d,np.sum((x-x[i])**2,1));d[selected]=-1
    return x[selected],radius


def pair(mesh,case_id=0,*,relative_radius=.18,count=32):
    c,r=centers(mesh,relative_radius=relative_radius,count=count);output=[]
    for quantity in ('u0','v0'):
        stream=np.random.default_rng(int(digest((VERSION,mesh.geometry_id,case_id,quantity,relative_radius,count))[:16],16))
        output.append(SmoothField(c,r,stream.standard_normal(count),mesh.geometry_id))
    return tuple(output)

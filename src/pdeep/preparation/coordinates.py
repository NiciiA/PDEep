"""Replay recorded draw arithmetic; accept ONLY an exact verified label SHA.

NumPy einsum can contract multiply-adds on one CPU and use separate rounded
operations on another. Seeds/triangles/weights are identical but bytes differ.
These are explicit IEEE754 evaluation orders of the ORIGINAL barycentric sum,
not rounding, nearest-neighbour matching, new sampling or reference evaluation.
"""
from functools import lru_cache
import ctypes,ctypes.util,itertools,math
import numpy as np
from ..core import require
from ..smooth.bank import physical_design
from ..smooth.definition import geometry,random_stream
from ..unified.streams import coordinates_id

@lru_cache(maxsize=1)
def _fma():
    if hasattr(math,'fma'):return math.fma
    library=ctypes.CDLL(ctypes.util.find_library('m') or None)
    operation=library.fma;operation.argtypes=[ctypes.c_double]*3;operation.restype=ctypes.c_double
    return operation

def arithmetic_variants(case,count,*,observation,set_id,design=None):
    """Same original counter recipe, with explicitly ordered barycentric sums."""
    design=physical_design(case,count,observation=observation,set_id=set_id) if design is None else design
    yield 'native',design
    mesh=geometry(case.family,.08);role='sensor' if observation else 'query'
    rng=random_stream(case.case_id,role,'position',set_id)
    chosen=rng.choice(len(mesh.cells),size=count,p=mesh.cell_areas/mesh.cell_areas.sum())
    uv=rng.random((count,2));uv[:,0]=np.sqrt(uv[:,0])
    w=np.column_stack((1-uv[:,0],uv[:,0]*(1-uv[:,1]),uv[:,0]*uv[:,1]));v=mesh.vertices[mesh.cells[chosen]];products=w[:,:,None]*v
    # Most common ARM producer order first after the consumer's native path.
    orders=[(2,1,0),*(o for o in itertools.permutations(range(3)) if o!=(2,1,0))]
    for fused in (True,False):
        for a,b,c in orders:
            if fused:
                fma=_fma();x=np.empty((count,2),np.float64)
                for i in range(count):
                    for j in range(2):x[i,j]=fma(float(w[i,a]),float(v[i,a,j]),fma(float(w[i,b]),float(v[i,b,j]),float(products[i,c,j])))
            else:x=(products[:,a]+products[:,b])+products[:,c]
            yield ('fma' if fused else 'separate')+f'-{a}{b}{c}',(x,*design[1:])

def prepared_design(dataset,case,count,*,observation=False,set_id=0):
    dataset.require_case(case)
    native=physical_design(case,count,observation=observation,set_id=set_id)
    if not hasattr(dataset.cache,'_payload'):return native # arithmetic-only unit providers
    labels=dataset.cache._payload(case)['labels'];extras=dataset.index.get('supplemental_labels',{}).get(case.case_id,{})
    for method,design in arithmetic_variants(case,count,observation=observation,set_id=set_id,design=native):
        key=coordinates_id(*design[:3])
        if key in labels or key in extras:
            profile=getattr(dataset,'load_profile',None)
            if profile is not None:
                methods=profile.setdefault('prepared_coordinate_replay',{});methods[method]=methods.get(method,0)+1
            return design
    from .loading import MissingPreparedDraw
    error=MissingPreparedDraw(dataset,case,*native[:3])
    error.request.update(role='observation' if observation else 'query',set_id=set_id,coordinate_replay='no supported arithmetic order matched a verified label key; never an approximate match')
    raise error

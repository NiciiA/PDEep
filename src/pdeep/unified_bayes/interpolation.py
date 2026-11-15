"""Observation-only graph-geodesic kernel interpolation; no shortcuts through holes."""
import time
import numpy as np
from scipy.sparse.csgraph import dijkstra
from ..core import require
from ..smooth.definition import geometry

class UnsupportedInterpolation(ValueError):pass


def geodesic_distances(mesh,sensors,queries,*,chunk=2048):
    si,_=mesh.locate(sensors);qi,_=mesh.locate(queries);sv=mesh.cells[si];qv=mesh.cells[qi]
    # Segments inside each endpoint triangle are in the represented domain.
    local_s=np.linalg.norm(sensors[:,None]-mesh.vertices[sv],axis=-1)
    unique,inverse=np.unique(sv,return_inverse=True);D=dijkstra(mesh.graph,directed=False,indices=unique)
    sensor_to_nodes=np.min(D[inverse.reshape(sv.shape)]+local_s[:,:,None],axis=1)
    output=[]
    for a in range(0,len(queries),chunk):
        vertices=qv[a:a+chunk];local_q=np.linalg.norm(queries[a:a+chunk,None]-mesh.vertices[vertices],axis=-1)
        distance=np.min(sensor_to_nodes[:,vertices]+local_q[None],axis=-1).T
        same=si[None]==qi[a:a+chunk,None]
        direct=np.linalg.norm(queries[a:a+chunk,None]-sensors[None],axis=-1)
        distance=np.where(same,np.minimum(distance,direct),distance);output.append(distance)
    return np.concatenate(output) if output else np.empty((0,len(sensors)))


def interpolate(pde,family,observed,query,values,sigma,*,spatial_length=.3,time_length=.2):
    """Fixed, unfitted bandwidths .3*sqrt(area), .2 physical T; inverse-noise kernel.

    Wave channels never mix. Geometry is an edge-graph approximation at canonical
    h=.08 plus within-triangle segments; this is NOT a PDE/boundary solver.
    No observations for a requested quantity is explicitly unsupported.
    """
    begin=time.perf_counter();mesh=geometry(family,.08);observed.validate(pde,family);query.validate(pde,family)
    y=np.asarray(values,dtype=np.float64);sigma=np.asarray(sigma,dtype=np.float64)
    require(y.shape==sigma.shape==(len(observed.position),) and np.isfinite(y).all() and np.isfinite(sigma).all() and (sigma>0).all(),'invalid interpolation measurements')
    D=geodesic_distances(mesh,observed.position,query.position);preprocessing=time.perf_counter()-begin;start=time.perf_counter();out=np.empty(len(query.position));L=np.sqrt(mesh.cell_areas.sum())
    for k in np.unique(query.quantity):
        rows=query.quantity==k;cols=observed.quantity==k
        if not cols.any():raise UnsupportedInterpolation('No available observations for quantity '+str(k))
        logw=-.5*(D[np.ix_(rows,cols)]/(spatial_length*L))**2-2*np.log(sigma[cols])[None]
        if pde!='poisson':logw-=.5*((query.physical_time[rows,None]-observed.physical_time[None,cols])/time_length)**2
        logw-=logw.max(axis=1,keepdims=True);w=np.exp(logw);w/=w.sum(axis=1,keepdims=True);out[rows]=w@y[cols]
    return out,{'preprocessing_seconds':preprocessing,'inference_seconds':time.perf_counter()-start,
                'assumptions':'graph-geodesic Gaussian kernel; per quantity, known noise weighting; no PDE, unknown inputs or fitted hyperparameters',
                'spatial_length_over_sqrt_area':spatial_length,'physical_time_length':None if pde=='poisson' else time_length}

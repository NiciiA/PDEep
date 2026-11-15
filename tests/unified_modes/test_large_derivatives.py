"""Independent FP32 coordinate finite differences on actual large wave networks."""
from pathlib import Path
import json
import numpy as np
import torch
from pdeep.smooth.data import Case
from pdeep.unified_modes.model import create
from pdeep.unified_modes.preflight import preflight_batch

def test_large_physical_space_time_derivatives(dataset):
    rows=[]
    for arch in ('A_GEO','C_GEO'):
        model=create(arch,dataset.norms,'large');case=Case('hole','wave','train',0,.12)
        batch=preflight_batch(dataset,case,'HYBRID',8,'cpu');ctx=model.prepare_context(batch.specs,batch.observations)[0]
        mesh=ctx.problem.mesh;tri=mesh.vertices[mesh.cells];perimeter=sum(np.linalg.norm(tri[:,i]-tri[:,(i+1)%3],axis=1) for i in range(3));radius=2*mesh.cell_areas/perimeter
        chosen=np.argsort(radius)[-5:];xy=tri[chosen].mean(1)
        # Stay inside the same finite element; the geometric C0/H1 interface is
        # intentionally not represented as globally C1 across mesh interfaces.
        h=min(2e-4,float(radius[chosen].min())/20)
        for quantity in ('u','v'):
            x=torch.tensor(xy,dtype=torch.float32,requires_grad=True);t=torch.linspace(.137,.731,len(x))[:,None].requires_grad_(True)
            y=model.evaluate(ctx,x,t,quantity);dx,dt=torch.autograd.grad(y.sum(),(x,t),retain_graph=True)
            checks=[]
            with torch.no_grad():
                for axis in range(3):
                    if axis<2:
                        delta=torch.zeros_like(x);delta[:,axis]=h
                        fd=(model.evaluate(ctx,x+delta,t,quantity)-model.evaluate(ctx,x-delta,t,quantity))/(2*h);actual=dx[:,axis:axis+1]
                    else:
                        fd=(model.evaluate(ctx,x,t+h,quantity)-model.evaluate(ctx,x,t-h,quantity))/(2*h);actual=dt
                    error=float((fd-actual).abs().max());scale=max(float(actual.abs().max()),dataset.norms.scale('wave',quantity),1e-6);tolerance=5e-3*scale
                    assert error<tolerance,(arch,quantity,axis,error,tolerance)
                    checks.append({'axis':('x','y','t')[axis],'maximum_absolute_error':error,'derivative_scale':scale,'tolerance':tolerance})
            rows.append({'architecture':arch,'quantity':quantity,'finite_difference_step':h,'checks':checks})
    p=Path('reports/g1_large/derivatives.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'device':'CPU FP32','scope':'independent central finite differences;5 element interiors in hole wave hybrid context; original decoder/geometry derivatives','rows':rows,'optimizer_steps':0},indent=2)+'\n')

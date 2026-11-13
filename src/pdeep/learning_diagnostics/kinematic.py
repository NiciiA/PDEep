"""Physical v=du/dt penalty. No solver, hidden targets or optimizer.

Counter keyed by the existing successful sampler cursor, so interrupted attempts
and exact resume reuse the same independent points. This is NOT supervised Q.
"""
from dataclasses import replace
import numpy as np
import torch
from ..core import require,digest
from ..smooth.definition import random_stream

REVISION='wave-kinematic-v1'

def coordinates(case,count=512,seed=42,cursor=0):
    require(case.pde=='wave' and count>=5,'kinematic points require wave and >=5 records')
    mesh=replace(case,conditioning_h=.08).problem.mesh
    x=mesh.sample(count,random_stream(REVISION,case.case_id,seed,cursor,'position'))
    rng=random_stream(REVISION,case.case_id,seed,cursor,'time')
    # Exactly 0 and 1, then equally allocated early/middle/late continuous times.
    n=(count-2)//3
    t=np.r_[0.,1.,rng.uniform(np.nextafter(0.,1.),.1,n),
            rng.uniform(np.nextafter(.1,1.),.5,n),rng.uniform(np.nextafter(.5,1.),1.,count-2-2*n)]
    rng.shuffle(t)
    return x,t


def physical_residual(model,context,x,t,*,create_graph=True):
    """Pointwise decoder contract is independently Jacobian/FD tested.

    evaluate returns physical units (including TRAIN RMS output scales). Time is
    the actual physical time; autograd includes any internal feature chain rule.
    """
    require(t.requires_grad,'physical time must require gradients')
    u=model.evaluate(context,x,t,'u');v=model.evaluate(context,x,t,'v')
    ut=torch.autograd.grad(u.sum(),t,create_graph=create_graph,retain_graph=True)[0]
    return ut-v


def kinematic_loss(model,contexts,normalizers,*,count=512,seed=42,cursor=0,chunk=512):
    require(chunk>0,'positive derivative chunk required');losses=[];reports=[]
    for context,case in contexts:
        x,t=coordinates(case,count,seed,cursor);pieces=[]
        for start in range(0,count,chunk):
            xx=torch.as_tensor(x[start:start+chunk],device=context.equation.device,dtype=context.equation.dtype)
            tt=torch.tensor(t[start:start+chunk,None],device=xx.device,dtype=xx.dtype,requires_grad=True)
            residual=physical_residual(model,context,xx,tt,create_graph=True)
            pieces.append((residual/normalizers.scale('wave','v')).square().sum()/count)
        losses.append(sum(pieces));reports.append({'records':count,'cursor':cursor,'seed':seed,'revision':REVISION,
            'identity':digest((case.case_id,seed,cursor,x.tolist(),t.tolist())),
            'normalization':'physical (du/dt-v) / frozen TRAIN velocity RMS; no offsets',
            'separate_from_supervised_queries':True,'time_counts':{'zero':int((t==0).sum()),'early':int(((t>0)&(t<=.1)).sum()),'middle':int(((t>.1)&(t<=.5)).sum()),'late':int(((t>.5)&(t<1)).sum()),'one':int((t==1).sum())}})
    return torch.stack(losses).mean(),reports

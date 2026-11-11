"""Bounded data/coordinate/error-budget audit, without neural optimizer calls."""
from pathlib import Path
import json,time
import numpy as np
import torch
from pdeep.core import digest,require
from pdeep.geometric.data import *
from pdeep.geometric.representation import layout,retrieval_weights

if __name__=='__main__':
    start=time.perf_counter();output=Path('reports/geometry_phase');n=Normalizers.fromdict(json.loads((output/'normalizers.json').read_text()));rows=[]
    # Resolve BOTH space and time independently; compare the SAME hidden field.
    for split,index in (('validation',0),('validation',3),('geometry_held_out',0),('geometry_held_out',1)):
        for pde in PDES:
            cs=[physical_case(split,pde,index,h=h,dt=.002) for h in (.08,.04,.02)]
            x=cs[-1].problem.mesh.sample(128,rng('reference_budget',split,index,pde));t=None if pde=='poisson' else np.full(len(x),.237)
            quantities=('u','v') if pde=='wave' else ('u',);errors={}
            for k in quantities:
                reference=cs[-1].evaluate(x,t,k);scale=n.scale(pde,k);vals=[c.evaluate(x,t,k) for c in cs]
                errors[k]=[{'h':c.reference['h'],'standardized_RMS_difference_to_h02':float(np.sqrt(np.mean((v-reference)**2))/scale),
                            'relative_L2_difference_to_h02':float(np.linalg.norm(v-reference)/np.linalg.norm(reference))} for c,v in zip(cs[:-1],vals[:-1])]
            require(len({c.case_id for c in cs})==1,'remeshing changed physical case')
            rows.append({'pde':pde,'split':split,'geometry':cs[0].problem.mesh.shape.family,'case_id':cs[0].case_id,'quantity_errors':errors,'geometry_budget':cs[0].reference['geometry_error']})
            physical_case.cache_clear()
    m=domain('hole',1.,.12);lay=layout(m,16,64)
    # One internal edge: affine interpolation has continuous trace, though the
    # normal derivative generally differs. Report that difference explicitly.
    all_edges=np.concatenate((m.cells[:,[0,1]],m.cells[:,[1,2]],m.cells[:,[2,0]]));edges,cnt=np.unique(np.sort(all_edges,1),axis=0,return_counts=True)
    edge=edges[cnt==2][len(edges[cnt==2])//2];a,b=m.vertices[edge];mid=(a+b)/2;direction=b-a;normal=np.array([direction[1],-direction[0]])/np.linalg.norm(direction);continuity=[]
    for eps in (1e-3,1e-4,1e-5,1e-6,1e-7):
        points=np.stack((mid+eps*normal,mid-eps*normal));x=torch.tensor(points,dtype=torch.float64,requires_grad=True);w=retrieval_weights(lay,x);weighted=(w*torch.arange(w.shape[1],dtype=x.dtype)[None]).sum(1)
        grad=torch.autograd.grad(weighted.sum(),x)[0]
        continuity.append({'epsilon':eps,'max_weight_jump':float((w[0]-w[1]).detach().abs().max()),'feature_jump':float((weighted[0]-weighted[1]).detach().abs()),'gradient_side_difference':float((grad[0]-grad[1]).abs().max()),'finite_gradients':bool(torch.isfinite(grad).all())})
    x=torch.tensor(m.sample(5,rng('coordinate-ad')),dtype=torch.float64,requires_grad=True);coeff=torch.arange(lay.distance2.shape[1],dtype=x.dtype);f=lambda x:(retrieval_weights(lay,x)*coeff).sum(-1)
    y=f(x);ad=torch.autograd.grad(y.sum(),x,create_graph=True)[0];errors=[]
    for d in range(2):
        offset=torch.zeros_like(x);offset[:,d]=1e-6;fd=(f(x+offset)-f(x-offset))/(2e-6);errors.append(float((fd-ad[:,d]).detach().abs().max()))
    hessian=torch.stack([torch.autograd.grad(ad[:,d].sum(),x,retain_graph=True)[0] for d in range(2)],1)
    seeds={split:{digest((VERSION,split,p,i,make_shape(*CATALOG[split][i%len(CATALOG[split])][:1],aspect=CATALOG[split][i%len(CATALOG[split])][1]).geometry_id)) for p in PDES for i in range(LIMITS[split])} for split in ('train','validation','test')}
    require(not (seeds['train']&seeds['validation'] or seeds['train']&seeds['test'] or seeds['validation']&seeds['test']),'physical split ID collision')
    result={'schema':VERSION,'reference_budget':rows,'retrieval':{'metric':'P1 interpolated squared mesh-graph distance; C0/H1','edge_continuity':continuity,'max_AD_FD_error':max(errors),'finite_piecewise_second_derivatives':bool(torch.isfinite(hessian).all()),'second_derivatives_at_interfaces':'not classically defined; not claimed'},
            'split_counts_all_PDEs':{k:len(v) for k,v in seeds.items()},'split_overlap':0,'normalizers':n.asdict(),'seconds':time.perf_counter()-start}
    max_velocity=max(row['quantity_errors']['v'][-1]['standardized_RMS_difference_to_h02'] for row in rows if row['pde']=='wave')
    result['dataset_release_gate']={'passed':False,'status':'BLOCKED','maximum_standardized_wave_velocity_difference':max_velocity,
        'reason':'Generated-field reference fidelity is not certified; h=.02 is only a comparator. Wave velocity differences exceed the sensor noise range. Stop before research training.'}
    (output/'data-and-retrieval.json').write_text(json.dumps(result,indent=2))
    print('Identity/retrieval audit complete; STOP: G3 generated-field fidelity BLOCKED.',result['seconds'],flush=True)
    raise SystemExit(2)

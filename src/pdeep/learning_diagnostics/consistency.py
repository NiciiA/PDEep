"""Independent wave diagnostics, using PHYSICAL coordinate derivatives.

Strong residuals are sampled element-interior diagnostics only: neither neural
P1 geometry conditioning nor the conforming P4 reference is globally C2. Weak
residuals use the unchanged common weak_loss and conditioning-mesh test functions.
The reference adapter evaluates the SAME P4/C1-Hermite cached function in Torch;
its coefficients/targets are used only by this diagnostic, never neural context.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import atomic_json,read_manifest,read_payload
from ..geometric.physics import weak_loss,WeakConfig
from .metrics import reconstruction
from .reference_cache import FixedReference

TIMES=(0.,.05,.1,.25,.5,.75,1.)

class ReferenceField:
    def __init__(self,cache):
        self.cache=cache;self.space=cache.reference.space
        self.context=SimpleNamespace(problem=cache.case.problem,equation=torch.zeros(1,dtype=torch.float64))
        self.free=np.full(len(self.space.position),-1,dtype=int);self.free[self.space.free]=np.arange(len(self.space.free))
    def evaluate(self,context,x,t,name):
        if name in self.cache.case.fields:
            f=self.cache.case.fields[name]
            centers=torch.as_tensor(f.centers,device=x.device,dtype=x.dtype)
            coefficients=torch.as_tensor(f.coefficients,device=x.device,dtype=x.dtype)
            r2=((x[:,None]-centers)**2).sum(-1)/f.radius**2
            return (f.amplitude*(1-r2).clamp_min(0).pow(4)@coefficients/np.sqrt(len(centers)))[:,None]
        s=self.space;ids,_=s.mesh.locate(x.detach().cpu().double().numpy());tri=s.mesh.vertices[s.mesh.cells[ids]]
        J=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
        inv=torch.as_tensor(np.linalg.inv(J),device=x.device,dtype=x.dtype);origin=torch.as_tensor(tri[:,0],device=x.device,dtype=x.dtype)
        local=torch.einsum('nij,nj->ni',inv,x-origin);bary=torch.cat((1-local.sum(-1,keepdim=True),local),-1)
        from ..wave_fidelity.reference import indices
        basis=[]
        for alpha in indices(s.degree):
            product=x[:,0]*0+1
            for j,a in enumerate(alpha):
                for k in range(a):product=product*(s.degree*bary[:,j]-k)/(a-k)
            basis.append(product)
        basis=torch.stack(basis,-1);nodes=self.free[s.dofs[ids]];valid=nodes>=0;nodes=nodes.clip(0)
        tt=t[:,0];dt=self.cache.dt;interval=np.minimum(np.floor(tt.detach().cpu().double().numpy()/dt).astype(int),len(self.cache.u)-2)
        require(((tt>=0)&(tt<=1)).all(),'reference diagnostic times outside [0,1]')
        z=(tt-torch.as_tensor(interval*dt,device=x.device,dtype=x.dtype))/dt
        def nodal(values,step):return torch.as_tensor(values[(interval+step)[:,None],nodes]*valid,device=x.device,dtype=x.dtype)
        ul,ur,vl,vr=nodal(self.cache.u,0),nodal(self.cache.u,1),nodal(self.cache.v,0),nodal(self.cache.v,1)
        if name=='u':state=(2*z**3-3*z*z+1)[:,None]*ul+(z**3-2*z*z+z)[:,None]*dt*vl+(-2*z**3+3*z*z)[:,None]*ur+(z**3-z*z)[:,None]*dt*vr
        elif name=='v':state=((6*z*z-6*z)/dt)[:,None]*ul+(3*z*z-4*z+1)[:,None]*vl+((-6*z*z+6*z)/dt)[:,None]*ur+(3*z*z-2*z)[:,None]*vr
        else:raise ValueError('wave diagnostic supports u/v and initial fields')
        return (basis*state).sum(-1,keepdim=True)


def independent_design(case,reference_space,seed=7193,count=256):
    """Area sampling, rejecting near edges of BOTH meshes, independent of truth."""
    mesh=case.problem.mesh;rng=np.random.default_rng(seed);points=[];n=0
    while n<count:
        x=mesh.sample(max(count,2*(count-n)),rng)
        _,b=mesh.locate(x);_,r=reference_space.mesh.locate(x)
        take=(b.min(-1)>1e-3)&(r.min(-1)>1e-3);selected=x[take][:count-n]
        points.append(selected);n+=len(selected)
    x=np.concatenate(points);times=np.tile(np.asarray(TIMES),len(x));xx=np.repeat(x,len(TIMES),axis=0)
    # Independent continuously sampled times, plus adequate coverage of every
    # named region. These are not the supervised query stream or evaluation grid.
    random_times=np.r_[rng.uniform(0,.1,count),rng.uniform(.1,.5,count),rng.uniform(.5,1,count)]
    return np.r_[xx,np.tile(x,(3,1))],np.r_[times,random_times]


def derivative_values(model,context,x,t,*,chunk=128):
    device=context.equation.device;dtype=context.equation.dtype;arrays={k:[] for k in ('u','v','kinematic','dynamic','ut','vt','laplacian')}
    for lo in range(0,len(x),chunk):
        xx=torch.tensor(x[lo:lo+chunk],device=device,dtype=dtype,requires_grad=True)
        tt=torch.tensor(t[lo:lo+chunk,None],device=device,dtype=dtype,requires_grad=True)
        u=model.evaluate(context,xx,tt,'u');v=model.evaluate(context,xx,tt,'v')
        ut=torch.autograd.grad(u.sum(),tt,create_graph=True,retain_graph=True)[0]
        vt=torch.autograd.grad(v.sum(),tt,create_graph=True,retain_graph=True)[0]
        dx=torch.autograd.grad(u.sum(),xx,create_graph=True,retain_graph=True)[0]
        lap=sum(torch.autograd.grad(dx[:,j].sum(),xx,retain_graph=True)[0][:,j:j+1] for j in (0,1))
        values={'u':u,'v':v,'ut':ut,'vt':vt,'laplacian':lap,'kinematic':ut-v,'dynamic':vt-context.problem.coefficient*lap}
        for name,val in values.items():arrays[name].append(val.detach().cpu().double().numpy()[:,0])
    return {name:np.concatenate(values) for name,values in arrays.items()}


def summarize(values,truth,t,case,norms):
    L=np.sqrt(case.problem.mesh.cell_areas.sum());su=norms.scale('wave','u');sv=norms.scale('wave','v')
    scales={'u':su,'v':sv,'kinematic':su+sv,'dynamic':sv+case.problem.coefficient*su/L**2}
    masks={'all':np.ones(len(t),bool),'t0':t==0,'early':(t>0)&(t<=.1),'middle':(t>.1)&(t<=.5),'late':t>.5,
           **{f't={v:g}':np.isclose(t,v,rtol=0,atol=1e-12) for v in TIMES}}
    result={}
    for name,take in masks.items():
        if not take.any():continue
        row={'records':int(take.sum())}
        for quantity,scale in scales.items():
            error=values[quantity]-truth[quantity] if quantity in {'u','v'} else values[quantity]
            rms=float(np.sqrt(np.mean(error[take]**2)));row[quantity]={'physical_RMS':rms,'standardized_RMS':rms/scale,'standardized_MSE':(rms/scale)**2}
        result[name]=row
    return {'scales':scales,'scale_definitions':'T=1; L=sqrt(area); kinematic su/T+sv; dynamic sv/T+c²su/L²; reconstruction quantity-specific fixed TRAIN RMS',
            'groups':result,'scope':'sampled element-interior strong residual; excludes distributional interface terms; NOT a global PDE certification'}


def reference_floor(data,*,cache=None,weak=WeakConfig(),count=128):
    cache=cache or data.cached_reference or FixedReference(data.case);model=ReferenceField(cache)
    x,t=independent_design(data.case,model.space,count=count)
    values=derivative_values(model,model.context,x,t)
    floor=weak_loss(model,[model.context],data.normalizers,weak)
    return {'pointwise':summarize(values,values,t,data.case,data.normalizers),
            'weak':{k:float(v.detach()) for k,v in floor.items() if k!='diagnostics'},'weak_definition':floor['diagnostics'],
            'reference_identity':cache.identity,'count':len(t),'dtype':'CPU float64'},cache


def evaluate_wave_checkpoints(checkpoints,certificate,root,*,device='auto',count=256,chunk=128):
    from .inspection import load_model
    from .data import prepare_data
    from ..smooth.training import select_device
    from .runtime import using,InputCache
    from ..artifact_tensors import state_identity
    root=Path(root);root.mkdir(parents=True,exist_ok=True);results={};missing={};references={}
    for label,path in checkpoints.items():
        if not path or not Path(path).expanduser().is_file():missing[label]='verified manifest unavailable; no model trained or substituted';continue
        try:
            model,meta=load_model(Path(path).expanduser(),certificate,device=device);c=meta['config']
            require(tuple(c['pdes'])==('wave',),'checkpoint is not a single wave diagnostic')
            require(meta.get('successful_update',0)>0,'wave evaluation requires actual trained updates')
            data=prepare_data(root/'fixed_data',certificate,experiment=c['experiment'],seed=c['root_seed'],pde='wave',family=c['training_families'][0],count=min(c['query_count'],1024),
                conditioning_h=c['conditioning_h'],grid_size=32,cache_reference=True,evaluation_times=TIMES)
            require(meta['fixed_case']['physical_case_id']==data.case.case_id,'checkpoint physical case differs from regenerated reference')
            from .data import token_id
            # Existing verified checkpoint's fixed-data artifact records the
            # sensor identity; alternatively verify the deterministic saved seed.
            _,saved_payload=read_payload(Path(path).expanduser(),required=('sampler',))
            expected_observations=saved_payload['sampler'].get('observations')
            require(expected_observations is None or expected_observations==token_id(data.observations),'checkpoint observation identity differs from diagnostic sensors')
            del saved_payload
            pair_id=digest((data.case.case_id,token_id(data.observations)));before=state_identity(model)
            cache=references.get(data.case.case_id) or data.cached_reference or FixedReference(data.case);references[data.case.case_id]=cache
            reference=ReferenceField(cache);x,t=independent_design(data.case,reference.space,count=count)
            truth=derivative_values(reference,reference.context,x,t,chunk=chunk)
            b=data.batch(select_device(device));runtime=InputCache()
            with using(runtime):
                context=model.prepare_context(b.specs,b.observations)[0]
                values=derivative_values(model,context,x,t,chunk=chunk)
                common=WeakConfig();physics=weak_loss(model,[context],data.normalizers,common)
            floor=weak_loss(reference,[reference.context],data.normalizers,WeakConfig())
            metrics=reconstruction(model,data)
            results[label]={'checkpoint':str(path),'artifact_id':meta['artifact_id'],'updates':meta['successful_update'],'pair_id':pair_id,'physical_case':data.case.case_id,
                'observation_identity':token_id(data.observations),'objective_variant':c['objective_variant'],'pointwise':summarize(values,truth,t,data.case,data.normalizers),
                'reference_floor':summarize(truth,truth,t,data.case,data.normalizers),'weak':{k:float(v.detach()) for k,v in physics.items() if k!='diagnostics'},
                'reference_weak':{k:float(v.detach()) for k,v in floor.items() if k!='diagnostics'},'reconstruction':metrics,'coordinates_identity':digest((x.tolist(),t.tolist())),
                'strong_records':len(t),'common_weak_config':{'spatial_order':3,'time_order':3,'tests':16,'chunk':256},'neural_dtype':str(context.equation.dtype),'neural_device':str(context.equation.device),'reference_dtype':'CPU float64','chunk':chunk,'optimizer_updates':0,'physical_coefficient_c_squared':data.case.problem.coefficient}
            require(before==state_identity(model),'evaluation changed trained model')
            np.savez_compressed(root/(digest(label)[:12]+'.npz'),position=x,time=t,**values,truth_u=truth['u'],truth_v=truth['v'])
            del context,model;runtime.clear_queries()
        except (ValueError,FileNotFoundError) as error:missing[label]=f'incompatible/unavailable: {error}'
    paired=len({r['pair_id'] for r in results.values()})<=1
    report={'models':results,'missing':missing,'comparison':'paired' if paired else 'NON-PAIRED: different physical cases/observations; compare within pair_id groups only','optimizer_updates':0}
    atomic_json(root/'wave-consistency.json',report)
    return report

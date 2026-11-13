"""Bounded nonoptimizing controls for the physical wave derivative pipeline.

Manufactured controls are independent of the FEM solver. Reference controls use
its full P4 element polynomial/C1 time representation, never a plotted image.
"""
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import atomic_json
from ..artifact_tensors import state_identity
from .consistency import derivative_values,ReferenceField,independent_design,reference_floor
from .kinematic import physical_residual

class ManufacturedWave:
    def __init__(self,Lx=1.,Ly=1.,c=.7,a=.7,b=-.3,T=1.,su=.13,sv=.83):
        self.Lx=Lx;self.Ly=Ly;self.c=c;self.a=a;self.b=b;self.T=T;self.su=su;self.sv=sv
        self.omega=c*np.pi*np.sqrt(Lx**-2+Ly**-2)
    def evaluate(self,context,x,t,name):
        # Deliberately normalize coordinates/time and standardize the heads, then
        # transform back: AD must include the full physical chain rule.
        xx=x/x.new_tensor([self.Lx,self.Ly]);tt=t/self.T;w=self.omega
        phi=torch.sin(np.pi*xx[:,0:1])*torch.sin(np.pi*xx[:,1:2])
        u=phi*(self.a*torch.cos(w*self.T*tt)+self.b*torch.sin(w*self.T*tt))
        v=w*phi*(-self.a*torch.sin(w*self.T*tt)+self.b*torch.cos(w*self.T*tt))
        return (u/self.su)*self.su if name=='u' else (v/self.sv)*self.sv


def _row(rows,name,error,scale=1.,rtol=2e-12,atol=2e-13,expected='exact; maximum absolute discrepancy'):
    error=float(error);tolerance=float(atol+rtol*scale)
    rows.append({'test':name,'expected':expected,'measured':error,'tolerance':tolerance,'passed':bool(np.isfinite(error) and error<=tolerance)})


def analytic_controls(count=31):
    rows=[];rng=np.random.default_rng(913)
    for label,Lx,Ly,c,T in [('square',1.,1.,.7,1.),('rectangle',1.2,.8,.43,2.7)]:
        model=ManufacturedWave(Lx,Ly,c,T=T);w=model.omega
        x=rng.uniform(.03,.97,(count,2))*[Lx,Ly];t=np.r_[0.,1.,rng.uniform(.001,.999,count-2)]
        ctx=SimpleNamespace(equation=torch.zeros(1,dtype=torch.float64),problem=SimpleNamespace(coefficient=c*c))
        values=derivative_values(model,ctx,x,t,chunk=11)
        phi=np.sin(np.pi*x[:,0]/Lx)*np.sin(np.pi*x[:,1]/Ly)
        F=model.a*np.cos(w*t)+model.b*np.sin(w*t)
        u=phi*F;v=w*phi*(-model.a*np.sin(w*t)+model.b*np.cos(w*t))
        lap=-np.pi**2*(Lx**-2+Ly**-2)*u
        for name,target in [('u',u),('v',v),('ut',v),('vt',-w*w*u),('laplacian',lap),('kinematic',np.zeros(count)),('dynamic',np.zeros(count))]:
            _row(rows,label+' '+name,np.max(abs(values[name]-target)),max(1.,np.max(abs(target))))
        xx=torch.tensor(x,dtype=torch.float64,requires_grad=True);tt=torch.tensor(t[:,None],dtype=torch.float64,requires_grad=True)
        uv=model.evaluate(ctx,xx,tt,'u');dx=torch.autograd.grad(uv.sum(),xx,create_graph=True)[0]
        ut=torch.autograd.grad(uv.sum(),tt,create_graph=True)[0];utt=torch.autograd.grad(ut.sum(),tt)[0]
        expected_dx=np.c_[np.pi/Lx*np.cos(np.pi*x[:,0]/Lx)*np.sin(np.pi*x[:,1]/Ly)*F,
                          np.pi/Ly*np.sin(np.pi*x[:,0]/Lx)*np.cos(np.pi*x[:,1]/Ly)*F]
        _row(rows,label+' physical spatial gradient',np.max(abs(dx.detach().numpy()-expected_dx)),np.max(abs(expected_dx)))
        _row(rows,label+' physical second time derivative',np.max(abs(utt.detach().numpy()[:,0]+w*w*u)),max(1,np.max(abs(w*w*u))))
        eps=1e-5
        with torch.no_grad():fd=(model.evaluate(ctx,xx,tt+eps,'u')-model.evaluate(ctx,xx,tt-eps,'u'))/(2*eps)
        _row(rows,label+' independent time FD',float((fd-ut).abs().max().detach()),float(ut.abs().max().detach()),rtol=5e-9,atol=1e-10)
        for quantity,target in [('u',u),('v',v)]:
            other=ManufacturedWave(Lx,Ly,c,T=1.,su=2.9,sv=.043).evaluate(ctx,xx,tt,quantity)
            _row(rows,label+' time/output normalization '+quantity,np.max(abs(other.detach().numpy()[:,0]-target)),max(1,np.max(abs(target))))
        boundary=np.r_[np.c_[np.zeros(4),rng.random(4)*Ly],np.c_[np.full(4,Lx),rng.random(4)*Ly],np.c_[rng.random(4)*Lx,np.zeros(4)],np.c_[rng.random(4)*Lx,np.full(4,Ly)]]
        bx=torch.tensor(boundary,dtype=torch.float64);bt=torch.full((16,1),.37,dtype=torch.float64)
        _row(rows,label+' homogeneous u/v boundary',max(float(model.evaluate(ctx,bx,bt,q).abs().max().detach()) for q in ('u','v')))
    return rows


def model_controls(model,data):
    """Actual unchanged decoder, including physical heads; no optimizer created."""
    rows=[];device=next(model.parameters()).device;b=data.batch(device);context=model.prepare_context(b.specs,b.observations)[0]
    mesh=data.case.problem.mesh;ids=np.linspace(0,len(mesh.cells)-1,6,dtype=int)
    xx=torch.tensor(mesh.vertices[mesh.cells[ids]].mean(1),device=device,dtype=torch.float32,requires_grad=True)
    tt=torch.tensor([[.071],[.193],[.327],[.513],[.697],[.863]],device=device,requires_grad=True)
    u=model.evaluate(context,xx,tt,'u');v=model.evaluate(context,xx,tt,'v')
    ut=torch.autograd.grad(u.sum(),tt,create_graph=True,retain_graph=True)[0]
    # Full small Jacobians prove sum-gradient cannot include other queries.
    jt=torch.stack([torch.autograd.grad(u[i,0],tt,retain_graph=True)[0][:,0] for i in range(len(tt))])
    jx=torch.stack([torch.autograd.grad(u[i,0],xx,retain_graph=True)[0] for i in range(len(tt))])
    eye=torch.eye(len(tt),device=device,dtype=torch.bool)
    _row(rows,'C decoder time Jacobian off-diagonal',float(jt[~eye].abs().max().detach()),atol=1e-9,rtol=0)
    _row(rows,'C decoder spatial Jacobian off-diagonal',float(jx[~eye].abs().max().detach()),atol=1e-9,rtol=0)
    _row(rows,'C sum-gradient versus Jacobian diagonal',float((ut[:,0]-jt.diag()).abs().max().detach()),float(ut.abs().max().detach()),rtol=2e-5,atol=2e-6)
    with torch.no_grad():
        singles=torch.cat([model.evaluate(context,xx[i:i+1],tt[i:i+1],'u') for i in range(len(xx))])
        _row(rows,'C individual/batched physical output',float((u-singles).abs().max().detach()),float(u.abs().max().detach()),rtol=3e-5,atol=2e-7)
        hidden=model._hidden(context,xx,tt)
        for name,head in [('u',model.u_head),('v',model.v_head)]:
            expected=head(hidden)*data.normalizers.scale('wave',name)
            _row(rows,'C physical '+name+' head/quantity identity',float((model.evaluate(context,xx,tt,name)-expected).abs().max().detach()),atol=1e-9,rtol=0)
        h=1e-4
        fd=(model.evaluate(context,xx,tt+h,'u')-model.evaluate(context,xx,tt-h,'u'))/(2*h)
        _row(rows,'C physical time AD/central FD',float((ut-fd).abs().max().detach()),float(ut.abs().max().detach()),rtol=.005,atol=3e-5)
    # Explicit spatial first-derivative FD, staying inside the same element.
    dx=torch.autograd.grad(u.sum(),xx,retain_graph=True)[0]
    with torch.no_grad():
        spatial=[];h=1e-4
        for k in (0,1):
            offset=torch.zeros_like(xx);offset[:,k]=h
            for sign in (-1,1):require(np.array_equal(mesh.locate((xx+sign*offset).cpu().double().numpy())[0],ids),'FD left conditioning element')
            spatial.append((model.evaluate(context,xx+offset,tt,'u')-model.evaluate(context,xx-offset,tt,'u'))/(2*h))
        _row(rows,'C physical spatial AD/central FD',float((dx-torch.cat(spatial,-1)).abs().max().detach()),float(dx.abs().max().detach()),rtol=.005,atol=3e-5)
    residual=physical_residual(model,context,xx,tt)
    g=torch.autograd.grad((residual/data.normalizers.scale('wave','v')).square().mean(),tuple(model.parameters()),allow_unused=True)
    gn=float(torch.sqrt(sum(a.detach().square().sum() for a in g if a is not None)))
    rows.append({'test':'C derivative parameter connection','expected':'finite positive gradient norm','measured':gn,'tolerance':None,'passed':bool(np.isfinite(gn) and gn>0)})
    # Compare derivative pipeline chunking, not just predictions.
    x=xx.detach().cpu().double().numpy();t=tt.detach().cpu().double().numpy()[:,0]
    a=derivative_values(model,context,x,t,chunk=6);bb=derivative_values(model,context,x,t,chunk=1)
    for key in a:_row(rows,'C derivative chunk '+key,np.max(abs(a[key]-bb[key])),max(1,np.max(abs(a[key]))),rtol=3e-4,atol=2e-5)
    if model.config.architecture=='A_GEO':
        for row in rows:row['test']=row['test'].replace('C ','A ',1)
    return rows


def reference_controls(data,*,count=12):
    cache=data.cached_reference
    if cache is None:
        from .reference_cache import FixedReference
        cache=FixedReference(data.case);data.cached_reference=cache
    reference=ReferenceField(cache);x,t=independent_design(data.case,reference.space,count=count)
    vals=derivative_values(reference,reference.context,x,t,chunk=64);rows=[]
    for quantity,k in [('u',0),('v',1)]:
        truth=cache.evaluate(x,t,np.full(len(t),k))
        _row(rows,'P4/Hermite '+quantity+' independent evaluator agreement',np.max(abs(vals[quantity]-truth)),max(1,np.max(abs(truth))),rtol=2e-11,atol=2e-12)
    sv=data.normalizers.scale('wave','v')
    _row(rows,'P4/Hermite exact kinematic identity',np.max(abs(vals['kinematic'])),sv,rtol=2e-10,atol=3e-11)
    # Independent FD inside one element and away from time integration knots.
    mesh=reference.space.mesh;centers=mesh.vertices[mesh.cells[:count]].mean(1);times=np.full(count,.23731)
    vv=derivative_values(reference,reference.context,centers,times,chunk=16)
    dt=1e-6;fd=(cache.evaluate(centers,times+dt)-cache.evaluate(centers,times-dt))/(2*dt)
    _row(rows,'P4/Hermite time AD/independent FD',np.max(abs(fd-vv['ut'])),max(sv,np.max(abs(vv['ut']))),rtol=5e-7,atol=1e-9)
    h=2e-6;lap=np.zeros(count)
    for k in (0,1):
        e=np.zeros_like(centers);e[:,k]=h
        for sign in (-1,1):require(np.array_equal(mesh.locate(centers+sign*e)[0],np.arange(count)),'reference FD left element')
        lap+=(cache.evaluate(centers+e,times)-2*cache.evaluate(centers,times)+cache.evaluate(centers-e,times))/h**2
    _row(rows,'P4 interior Laplacian AD/independent FD',np.max(abs(lap-vv['laplacian'])),max(1,np.max(abs(vv['laplacian']))),rtol=3e-4,atol=2e-5)
    floor,_=reference_floor(data,cache=cache,count=count)
    for name in ('weak','initial','boundary'):
        value=floor['weak'][name]
        rows.append({'test':'certified reference '+name+' floor','expected':'finite nonnegative floor; continuum zero NOT required','measured':value,'tolerance':None,'passed':bool(np.isfinite(value) and value>=0)})
    # Initial finite-element projection versus continuous initial fields is an
    # approximation floor, not a demand for bit-identical nodal interpolation.
    at0=derivative_values(reference,reference.context,centers,np.zeros(count))
    floor['initial_field_physical_RMSE']={q:float(np.sqrt(np.mean((at0[q]-data.case.fields[q+'0'](centers))**2))) for q in ('u','v')}
    floor['kinematic_sv_normalized_MSE']=float(np.mean((vals['kinematic']/sv)**2))
    floor['strong_interior_dynamic_physical_RMS']=float(np.sqrt(np.mean(vals['dynamic']**2)))
    floor['strong_scope']='element interiors only; excluded facet distributional terms; this is not continuum error or recertification'
    return rows,floor


def verify_wave(plan,*,data=None,model=None,reference_count=12):
    """CPU64 analytic/reference, selected-device actual decoder; zero updates."""
    from .data import prepare_data
    from ..geometric.models import create
    from ..smooth.training import select_device
    from .engine import diagnostic_source
    started=time.perf_counter();require(plan.model in {'A_GEO','C_GEO'} and plan.recipes[0]['pde']=='wave','A/C wave verification only')
    require(plan.source==diagnostic_source(),'stale wave plan; reload code and prepare again')
    data=data or prepare_data(Path(plan.root)/'learning_diagnostics'/'fixed_data',plan.certificate,experiment=plan.experiments[0],seed=plan.settings.seed,pde='wave',family='square',
        count=min(plan.settings.query_count,1024),conditioning_h=plan.settings.advanced.get('conditioning_h',.12),grid_size=plan.settings.advanced.get('grid_size',64),cache_reference=True,evaluation_times=(0.,.05,.1,.25,.5,.75,1.))
    model=model or create(plan.model,data.normalizers,plan.capacity,seed=plan.settings.seed,device=select_device(plan.settings.device))
    before=state_identity(model);rows=analytic_controls()+model_controls(model,data);rr,floor=reference_controls(data,count=reference_count);rows+=rr
    gradients=None
    if 'KIN_001' in plan.settings.variant_settings:
        from .wave_study import member_config
        from .study_controls import configured_adapter
        from .metrics import gradient_terms
        cfg=member_config(plan,data.identity,'KIN_001','nonoptimizing-wave-derivative-probe')
        adapter=configured_adapter(cfg)
        gradients=gradient_terms(model,data.batch(cfg.device),adapter)
        rows.append({'test':'KIN raw/weighted gradient finite and positive','expected':'parameter connected, lambda applied exactly',
            'measured':gradients['raw_gradient_norms']['kinematic'],'tolerance':None,
            'passed':bool(gradients['raw_gradient_norms']['kinematic']>0 and np.isfinite(gradients['raw_gradient_norms']['kinematic']) and np.isclose(gradients['weighted_gradient_norms']['kinematic'],cfg.kinematic_weight*gradients['raw_gradient_norms']['kinematic'],rtol=2e-5))})
    require(before==state_identity(model),'derivative verification changed model weights')
    report={'passed':all(r['passed'] for r in rows),'gradient_scale_diagnostics':gradients,'rows':rows,'reference_floor':floor,'plan_identity':plan.identity,'source':diagnostic_source(),
        'certificate':plan.certificate['certification_id'],'physical_case':data.case.case_id,'data_identity':data.identity,'observation_identity':data.identities['observation'],'evaluation_identity':data.identities['diagnostic'],
        'initialization_identity':before,'model_config':asdict(model.config),'model_state':'fresh deterministic initialization, NOT a trained-model result',
        'device':str(next(model.parameters()).device),'dtype':str(next(model.parameters()).dtype),'reference_dtype':'CPU float64','optimizer_updates':0,'qualification_executed':False,
        'elapsed_seconds':time.perf_counter()-started,'CUDA_device_name':torch.cuda.get_device_name(next(model.parameters()).device) if next(model.parameters()).device.type=='cuda' else None,'CUDA_memory_preflight_executed':False,
        'decoder_contract':'query independent given shared context; LayerNorm only over features; no query-to-query attention; full small time/spatial Jacobians tested',
        'legacy_reporting':'existing kinematic standardized RMS uses su+sv; this study separately reports training/kinematic_sv_normalized_MSE with sv only'}
    atomic_json(Path(plan.root)/'learning_diagnostics'/'wave_kinematic'/'derivative-verification.json',report)
    return report


def display_verification(report):
    from IPython.display import display,Markdown
    rows=['|Test|Expected|Measured max error/floor|Tolerance|Pass|','|---|---|---:|---:|---|']
    for row in report['rows']:
        rows.append(f"|{row['test']}|{row['expected']}|{row['measured']:.5g}|{row['tolerance'] if row['tolerance'] is not None else 'floor/positive norm'}|{'PASS' if row['passed'] else 'FAIL'}|")
    display(Markdown('\n'.join(rows)))
    gradients=report.get('gradient_scale_diagnostics')
    if gradients:
        print('Fixed-batch gradient norms (field, auxiliary, kinematic): raw',gradients['raw_gradient_norms'],'weighted',gradients['weighted_gradient_norms'])
        print('Gradient cosines:',gradients['weighted_gradient_cosines'])
        if gradients['kinematic_gradient_dominates_100x']:print('FLAG: weighted kinematic gradient >=100x field. Declared weight retained; inspect before long training.')
    print(f"Derivative gate: {'PASS' if report['passed'] else 'BLOCKED'} · zero optimizer updates · {report['elapsed_seconds']:.1f}s · {report['device']}/{report['dtype']} · reference CPU64; no recertification.")

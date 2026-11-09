"""Independent interior strong checks, boundary and inferred-initial diagnostics.

Classical derivatives are appropriate only for this smooth modal V1 benchmark.
No residual is replaced by zero merely because a method uses a physical solver.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from .batches import QueryBatch
from .reference import evaluate, eigenvalues
from .constraints import field_derivatives, strong_diagnostics, physical_losses
from .representations import float32_batch
from .evaluation_metrics import errors


def diagnostic_query(plan, spec):
    pts=torch.tensor(plan.diagnostic_points,dtype=torch.float64)
    valid=torch.ones(1,len(pts),dtype=torch.bool)
    return QueryBatch(pts[None,:,:2].clone(),pts[None,:,2:].clone() if spec.name!='poisson' else torch.zeros(1,len(pts),1,dtype=torch.float64),
                      valid & (spec.name!='poisson'),torch.zeros_like(valid,dtype=torch.int64),valid)

class StateAdapter:
    def __init__(self, session):self.s=session
    def evaluate_state(self,ctx,q):
        m=self.s.method
        if m.architecture=='C':return m.model.evaluate_state(ctx,q)
        if m.architecture=='A':
            u=m.model.predict_queries(ctx,replace(q,quantity_id=torch.zeros_like(q.quantity_id))).predicted_value
            v=m.model.predict_queries(ctx,replace(q,quantity_id=torch.ones_like(q.quantity_id))).predicted_value if self.s.spec.name=='wave' else u*0
            return SimpleNamespace(u=u,v=v+q.physical_time*0)
        time=None if self.s.spec.name=='poisson' else q.physical_time
        u=evaluate(self.s.spec.name,self.s.xi,q.position,time,quantity='u')+q.physical_time*0
        v=evaluate(self.s.spec.name,self.s.xi,q.position,time,quantity='v') if self.s.spec.name=='wave' else u*0
        return SimpleNamespace(u=u,v=v)
    def infer_unknowns(self,ctx):
        if self.s.method.architecture in {'A','C'}:return self.s.method.model.infer_unknowns(ctx)
        xi=torch.nn.functional.pad(self.s.xi,(0,18-len(self.s.xi)))[None]
        return SimpleNamespace(xi_hat=xi)


def physical_diagnostics(session, plan, truth, normalizers):
    spec=session.spec;method=session.method; adapter=StateAdapter(session)
    q64=diagnostic_query(plan,spec); q=q64
    neural=method.architecture in {'A','C'}
    if neural:
        device=method.device;q=float32_batch(q64,device=device.type if device.type=='mps' else str(device))
        context=session.context
    else:context=SimpleNamespace(specs=(spec,))
    with torch.inference_mode(False),torch.enable_grad():
        strong=strong_diagnostics(adapter,context,q)[0]
        result={k:v for k,v in strong.items() if k!='residual_values'}
        result['max_abs_residual']=[float(v.abs().max()) for v in strong['residual_values']]
        if not neural:
            query=replace(q64,position=q64.position.clone().requires_grad_(),physical_time=q64.physical_time.clone().requires_grad_())
            d=field_derivatives(adapter,context,query,create_graph=False,second=True)
            t=None if spec.name=='poisson' else q64.physical_time
            dx=evaluate(spec.name,session.xi,q64.position,t,spatial_derivative=(2,0))
            dy=evaluate(spec.name,session.xi,q64.position,t,spatial_derivative=(0,2))
            result['independent_AD_vs_analytic_max']={'laplacian':float((d.laplacian_u-dx-dy).detach().abs().max())}
            if t is not None:
                ut=evaluate(spec.name,session.xi,q64.position,t,time_order=1)
                result['independent_AD_vs_analytic_max']['u_t']=float((d.u_t-ut).detach().abs().max())
                if spec.name=='wave':
                    vt=evaluate(spec.name,session.xi,q64.position,t,quantity='v',time_order=1)
                    result['independent_AD_vs_analytic_max']['v_t']=float((d.v_t-vt).detach().abs().max())
        if method.architecture=='C':
            weak=physical_losses(method.model,session.context,create_graph=False)
            result['C_separate_weak_diagnostics']={'standardized_weak_MSE':float(weak.weak_per_case.detach()[0]),
                   'standardized_inferred_initial_MSE':float(weak.initial_per_case.detach()[0]),
                   'physical_weak_MSE':[float(v.detach()) for v in weak.cases[0]['physical_weak_MSE']],
                   'quadrature':'canonical 16x16 spatial, 8 physical-time Gauss points; separate from strong points'}
    # Four sides and t=0/early/interior/end for dynamic families; stationary time absent.
    axis=torch.linspace(0,1,17,dtype=torch.float64);z=torch.zeros_like(axis);o=torch.ones_like(axis)
    p=torch.cat((torch.stack((axis,z),1),torch.stack((axis,o),1),torch.stack((z,axis),1),torch.stack((o,axis),1)))
    times=(0.,) if spec.name=='poisson' else (0.,.02,.4,1.)
    p=p.repeat(len(times),1)[None];t=torch.tensor(times,dtype=torch.float64).repeat_interleave(68)[None,:,None]
    valid=torch.ones(p.shape[:2],dtype=torch.bool)
    bq=QueryBatch(p,t,valid&(spec.name!='poisson'),torch.zeros_like(valid,dtype=torch.int64),valid)
    result['boundary_max_abs']={}
    for quantity in ((0,1) if spec.name=='wave' else (0,)):
        b=replace(bq,quantity_id=torch.full_like(valid,quantity,dtype=torch.int64))
        result['boundary_max_abs']['v' if quantity else 'u']=float(session.predict(b).abs().max())
    result['initial']={}
    if spec.name!='poisson':
        init=replace(q64,physical_time=torch.zeros_like(q64.physical_time))
        for quantity in ((0,1) if spec.name=='wave' else (0,)):
            iq=replace(init,quantity_id=torch.full_like(init.quantity_id,quantity))
            pred=session.predict(iq); kind='v' if quantity else 'u'
            target=evaluate(spec.name,truth,iq.position,iq.physical_time,quantity=kind)
            inferred=evaluate(spec.name,session.xi,iq.position,iq.physical_time,quantity=kind)
            result['initial'][kind]={'hidden_truth_error':errors(pred,target,normalizers.scale(spec.name,quantity)),
                                      'inferred_initial_consistency':errors(pred,inferred,normalizers.scale(spec.name,quantity))}
    return result

"""Independent pre-training verification of the actual constraint contractions."""
from dataclasses import replace
import time
import torch
from .constraints import (Quadrature,weak_integrals,residual_scales,known_scalar,prior_initial_scales)
from .data import CaseDataset
from .reference import canonical_spec,evaluate,basis,basis_gradient,eigenvalues


def mathematical_constraint_checks():
    began=time.perf_counter();checks=[];orders={}
    def record(name,error,tolerance):
        error=float(error.detach()) if isinstance(error,torch.Tensor) else float(error)
        checks.append({"name":name,"max_abs_error":error,"tolerance":tolerance,"passed":error<=tolerance})
    for order in (16,32):
        rule=Quadrature(order);w=rule.spatial_weights;phi=rule.test_values;dp=rule.test_gradients
        record(f"spatial_weights_{order}",(w.sum()-1).abs(),2e-15)
        record(f"temporal_weights_{order}",(rule.time_weights.sum()-1).abs(),2e-15)
        record(f"basis_mass_{order}",(phi.T@(w[:,None]*phi)-torch.eye(9,dtype=torch.float64)).abs().max(),1e-11)
        stiffness=torch.einsum('sid,sjd,s->ij',dp,dp,w)
        record(f"basis_stiffness_{order}",(stiffness-torch.diag(rule.eigenvalues)).abs().max(),1e-9)
        residuals={}
        for name in ("poisson","heat","wave"):
            spec=canonical_spec(name);xi=CaseDataset().case("train",name,11).tensor();xi=torch.cat((xi,torch.zeros(18-len(xi),dtype=torch.float64)))
            T=1 if name=="poisson" else 8;S=len(w)
            p=rule.spatial_points[None].expand(T,-1,-1).clone().requires_grad_()
            t=(torch.zeros(1,dtype=torch.float64) if name=="poisson" else rule.time_nodes)[:,None,None].expand(-1,S,1).clone().requires_grad_()
            kwargs={"physical_time":None if name=="poisson" else t}
            u=evaluate(name,xi[:9] if name!="wave" else xi,p,**kwargs)
            grad=torch.autograd.grad(u.sum(),p,create_graph=True)[0]
            ut=torch.zeros_like(u) if name=="poisson" else torch.autograd.grad(u.sum(),t,retain_graph=True)[0]
            if name=="wave":
                v=evaluate(name,xi,p,t,quantity="v");vt=torch.autograd.grad(v.sum(),t)[0]
            else:v=vt=torch.zeros_like(u)
            actual=weak_integrals(spec,rule,u=u,v=v,grad_u=grad,u_t=ut,v_t=vt,xi=xi)
            residuals[name]=torch.cat(actual,-1).detach()
            scale=max(1.,*(float(s.max()) for s in residual_scales(spec,rule)))
            record(f"exact_{name}_weak_{order}",residuals[name].abs().max(),1e-11+1e-10*scale)
            # Independent analytic derivatives, not another weak formula copy.
            analytic_grad=torch.stack([evaluate(name,xi[:9] if name!="wave" else xi,p,**kwargs,spatial_derivative=d)[...,0] for d in ((1,0),(0,1))],-1)
            record(f"reference_spatial_AD_{name}_{order}",(grad-analytic_grad).abs().max(),1e-11)
            if name!="poisson":
                analytic_ut=evaluate(name,xi[:9] if name=="heat" else xi,p,t,time_order=1)
                record(f"reference_time_AD_{name}_{order}",(ut-analytic_ut).abs().max(),1e-11)
            # Controlled mode-1 field perturbation, independently expected weak response.
            epsilon=.13
            perturbed=weak_integrals(spec,rule,u=u+epsilon*phi[:,0][None,:,None],v=v,grad_u=grad+epsilon*dp[:,0][None],u_t=ut,v_t=vt,xi=xi)
            response=perturbed[-1]-actual[-1]
            expected=torch.zeros_like(response)
            factor=1. if name=="poisson" else known_scalar(spec,"alpha") if name=="heat" else known_scalar(spec,"c")**2
            expected[:,0]=epsilon*factor*rule.eigenvalues[0]
            record(f"field_perturbation_response_{name}_{order}",(response-expected).abs().max(),1e-9)
            if name=="poisson":
                altered=xi.clone();altered[0]+=epsilon
                source=weak_integrals(spec,rule,u=u,v=v,grad_u=grad,u_t=ut,v_t=vt,xi=altered)[0]-actual[0]
                record(f"source_perturbation_response_{order}",(source+expected).abs().max(),1e-9)
            else:
                # Exact initial state versus a separately perturbed inferred initial input.
                rms=prior_initial_scales(spec)[0]
                discrepancy=((epsilon*phi[:,0]/rms).square()*w).sum()
                record(f"initial_perturbation_response_{name}_{order}",(discrepancy-(epsilon/rms)**2).abs(),1e-12)
            if name=="wave":
                altered=weak_integrals(spec,rule,u=u,v=v+epsilon*phi[:,0][None,:,None],grad_u=grad,u_t=ut,v_t=vt,xi=xi)
                expected_v=torch.zeros_like(altered[0]);expected_v[:,0]=-epsilon
                record(f"wave_velocity_perturbation_{order}",(altered[0]-actual[0]-expected_v).abs().max(),1e-11)
                vrms=prior_initial_scales(spec)[1]
                record(f"wave_velocity_initial_perturbation_{order}",(((epsilon*phi[:,0]/vrms).square()*w).sum()-(epsilon/vrms)**2).abs(),1e-12)
            # Small reference finite difference with scale-aware error.
            points=torch.tensor([[.211,.379],[.481,.613]],dtype=torch.float64);h=1e-5
            plus=points.clone();minus=points.clone();plus[:,0]+=h;minus[:,0]-=h
            physical_time=None if name=="poisson" else torch.full((2,1),.173,dtype=torch.float64)
            coeff=xi if name=="wave" else xi[:9]
            fd=(evaluate(name,coeff,plus,physical_time)-evaluate(name,coeff,minus,physical_time))/(2*h)
            dx=evaluate(name,coeff,points,physical_time,spatial_derivative=(1,0))
            record(f"reference_FD_{name}_{order}",((fd-dx).abs()/(1+dx.abs())).max(),1e-7)
        orders[order]=residuals
    for name in ("poisson","heat","wave"):
        record(f"exact_16_vs_32_{name}",(orders[16][name]-orders[32][name]).abs().max(),1e-8)
    return {"passed":all(c["passed"] for c in checks),"checks":checks,"device":"cpu","dtype":"float64",
            "spatial_orders":[16,32],"time_order":8,"elapsed_seconds":time.perf_counter()-began,
            "limitations":"Nine tests detect only their projections; a mode orthogonal to them need not be detected. Neural quadrature accuracy is measured separately, never assumed."}

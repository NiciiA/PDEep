"""V1 physical constraints: fixed quadrature, nine tests, AD and explicit forms.

No field solver. Basis values/gradients are cached CPU-float64 reference constants,
explicitly copied to neural device/precision. Neural derivatives stay on device.
Spatial chunk integrals are assembled BEFORE squaring. All eight physical times
are used. Initial t=0 is separate from interior temporal quadrature.
"""
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
import torch
from torch import nn
from .batches import QueryBatch
from .core import require
from .reference import basis,basis_gradient,eigenvalues
from .representation_checks import sync


class Quadrature(nn.Module):
    def __init__(self,spatial_order=16,time_order=8):
        super().__init__()
        require(type(spatial_order) is int and spatial_order>=2 and type(time_order) is int and time_order>=1,"invalid Gauss-Legendre orders")
        self.spatial_order=spatial_order;self.time_order=time_order
        x,w=np.polynomial.legendre.leggauss(spatial_order)
        x=torch.from_numpy((x+1)/2);w=torch.from_numpy(w/2)
        yy,xx=torch.meshgrid(x,x,indexing="ij")
        points=torch.stack((xx,yy),-1).reshape(-1,2)
        weights=(w[:,None]*w[None,:]).flatten()
        t,tw=np.polynomial.legendre.leggauss(time_order)
        for name,value in (("spatial_points",points),("spatial_weights",weights),("test_values",basis(points)),
                           ("test_gradients",basis_gradient(points)),("eigenvalues",eigenvalues()),
                           ("time_nodes",torch.from_numpy((t+1)/2)),("time_weights",torch.from_numpy(tw/2))):self.register_buffer(name,value)


@lru_cache(maxsize=12)
def fixed_quadrature(spatial_order=16,time_order=8,device="cpu",dtype=torch.float32):
    require(dtype in (torch.float32,torch.float64),"constraint precision must be explicit float32/64")
    require(not (torch.device(device).type=="mps" and dtype==torch.float64),"MPS float64 constraints unsupported; no fallback")
    # Never mutate returned fixed buffers. No trainable state or neural derivative is cached.
    return Quadrature(spatial_order,time_order).to(device=device,dtype=dtype)


def known_scalar(spec,name):
    matches=[b for b in spec.bindings if b.id==name]
    require(len(matches)==1 and matches[0].value is not None and matches[0].value>0,f"known positive {name} required")
    return matches[0].value


def prior_initial_scales(spec):
    """Exact training-prior RMS via orthonormality; no held-out calibration.

    E integral u0² = sum_j(mean_j²+std_j²); unit area. These differ from
    time-averaged dynamic quantity RMS and are persisted in C exports.
    """
    return tuple(sum(m*m+s*s for m,s in zip(b.prior.mean,b.prior.scale))**.5 for b in spec.parameter_blocks)


def residual_scales(spec,rule):
    tau=rule.eigenvalues.new_tensor(spec.parameter_blocks[0].prior.scale)
    lam=rule.eigenvalues
    if spec.name=="poisson":return (lam*tau,)
    if spec.name=="heat":return (known_scalar(spec,"alpha")*lam*tau,)
    require(spec.name=="wave","unsupported weak family")
    omega=known_scalar(spec,"c")*lam.sqrt()
    return omega*tau,omega.square()*tau


def modal_input(xi,rule,*,kind):
    """Source or inferred initial field, NOT a solution/evolution operator."""
    require(xi.shape[-1]==18,"modal input requires canonical 18 padded slots")
    if kind=="source":coefficients=xi[...,:9]*rule.eigenvalues
    elif kind=="u0":coefficients=xi[...,:9]
    elif kind=="v0":coefficients=xi[...,9:18]
    else:raise ValueError("unknown modal input kind")
    return coefficients@rule.test_values.T


def weak_integrals(spec,rule,*,u,v,grad_u,u_t,v_t,xi,weights=None,test_values=None,test_gradients=None):
    """One reusable contraction for reference and neural evaluation.

    Fields [T,S,1], grad_u [T,S,2], xi [18]. Optional spatial slices let
    callers accumulate partial integrals; this function NEVER squares them.
    """
    w=rule.spatial_weights if weights is None else weights
    phi=rule.test_values if test_values is None else test_values
    dphi=rule.test_gradients if test_gradients is None else test_gradients
    stiffness=torch.einsum("tsd,sjd,s->tj",grad_u,dphi,w)
    def project(value):return torch.einsum("ts,sj,s->tj",value[...,0],phi,w)
    if spec.name=="poisson":
        source=(xi[:9]*rule.eigenvalues)@phi.T
        return (stiffness-torch.einsum("s,sj,s->j",source,phi,w)[None],)
    if spec.name=="heat":return (project(u_t)+known_scalar(spec,"alpha")*stiffness,)
    require(spec.name=="wave","unsupported weak family")
    return project(u_t-v),project(v_t)+known_scalar(spec,"c")**2*stiffness


@dataclass(frozen=True)
class FieldDerivatives:
    u: torch.Tensor
    v: torch.Tensor
    grad_u: torch.Tensor
    u_t: torch.Tensor
    v_t: torch.Tensor
    laplacian_u: torch.Tensor | None
    forward_seconds: float
    derivative_seconds: float


def field_derivatives(model,context,query,*,create_graph=True,second=False):
    """Differentiate the pointwise neural field in physical coordinates.

    Caller owns coordinates and enables requires_grad. No detach, no no_grad,
    and no implicit CPU movement. Sum trick is tested against single-point AD.
    """
    require(torch.is_grad_enabled(),"coordinate differentiation requires gradient tracking")
    require(query.position.requires_grad and query.physical_time.requires_grad,"physical position/time storage requires_grad required")
    device=query.position.device;sync(device);start=time.perf_counter()
    state=model.evaluate_state(context,query);sync(device);forward=time.perf_counter()-start
    start=time.perf_counter()
    grad_u,u_t=torch.autograd.grad(state.u.sum(),(query.position,query.physical_time),create_graph=create_graph or second,retain_graph=True)
    v_t=torch.autograd.grad(state.v.sum(),query.physical_time,create_graph=create_graph,retain_graph=True)[0]
    lap=None
    if second:
        lap=sum(torch.autograd.grad(grad_u[...,d].sum(),query.position,create_graph=create_graph,retain_graph=True)[0][...,d:d+1] for d in (0,1))
    sync(device)
    return FieldDerivatives(state.u,state.v,grad_u,u_t,v_t,lap,forward,time.perf_counter()-start)


def quadrature_query(spec,rule,*,initial=False,lo=0,hi=None):
    s=len(rule.spatial_points);hi=s if hi is None else hi
    p=rule.spatial_points[lo:hi]
    t=rule.time_nodes if spec.name!="poisson" and not initial else rule.time_nodes.new_zeros(1)
    position=p[None].expand(len(t),-1,-1).reshape(1,-1,2).clone().requires_grad_()
    time=t[:,None,None].expand(-1,len(p),1).reshape(1,-1,1).clone().requires_grad_()
    valid=torch.ones(position.shape[:2],device=position.device,dtype=torch.bool)
    return QueryBatch(position,time,valid&(spec.name!="poisson"),torch.zeros_like(valid,dtype=torch.int64),valid)


@dataclass(frozen=True)
class PhysicalLosses:
    weak_per_case: torch.Tensor
    initial_per_case: torch.Tensor
    cases: list
    timing: dict


def physical_losses(model,context,*,xi_hat=None,spatial_order=16,time_order=8,spatial_chunk=None,create_graph=True):
    require(time_order==8,"V1 weak objective requires all eight physical-time nodes")
    dtype=context.conditioning.dtype;device=context.conditioning.device
    rule=fixed_quadrature(spatial_order,time_order,str(device),dtype)
    xi=model.infer_unknowns(context).xi_hat if xi_hat is None else xi_hat
    require(xi.shape==(len(context.specs),18),"constraint xi shape mismatch")
    size=len(rule.spatial_points);chunk=size if spatial_chunk is None else spatial_chunk
    require(type(chunk) is int and chunk>0,"positive spatial chunk required")
    weak=[];initial=[];details=[];timing={"quadrature_forward_seconds":0.,"coordinate_derivative_seconds":0.,"field_evaluations":0,"weak_points":0,"initial_points":0}
    for row,spec in enumerate(context.specs):
        ctx=context.select(row);integrals=None
        for lo in range(0,size,chunk):
            hi=min(size,lo+chunk);q=quadrature_query(spec,rule,lo=lo,hi=hi)
            derivative=field_derivatives(model,ctx,q,create_graph=create_graph)
            T=8 if spec.name!="poisson" else 1;S=hi-lo
            kwargs={name:getattr(derivative,name).reshape(T,S,-1) for name in ("u","v","grad_u","u_t","v_t")}
            part=weak_integrals(spec,rule,**kwargs,xi=xi[row],weights=rule.spatial_weights[lo:hi],test_values=rule.test_values[lo:hi],test_gradients=rule.test_gradients[lo:hi])
            integrals=part if integrals is None else tuple(a+b for a,b in zip(integrals,part))
            timing["quadrature_forward_seconds"]+=derivative.forward_seconds;timing["coordinate_derivative_seconds"]+=derivative.derivative_seconds
            timing["field_evaluations"]+=1;timing["weak_points"]+=T*S
        scales=residual_scales(spec,rule)
        terms=[(r/s).square().mean(-1) for r,s in zip(integrals,scales)]
        tw=rule.time_weights if spec.name!="poisson" else rule.time_weights.new_ones(1)
        loss=sum((term*tw).sum() for term in terms)/len(terms)
        weak.append(loss)
        init=xi[row].sum()*0
        initial_difference=[]
        if spec.name!="poisson":
            q=quadrature_query(spec,rule,initial=True)
            sync(device);start=time.perf_counter();state=model.evaluate_state(ctx,q);sync(device)
            timing["quadrature_forward_seconds"]+=time.perf_counter()-start;timing["field_evaluations"]+=1;timing["initial_points"]+=size
            rms=prior_initial_scales(spec)
            initial_difference=[state.u[0,:,0]-modal_input(xi[row],rule,kind="u0")]
            if spec.name=="wave":initial_difference.append(state.v[0,:,0]-modal_input(xi[row],rule,kind="v0"))
            init=sum((((diff/scale)**2)*rule.spatial_weights).sum() for diff,scale in zip(initial_difference,rms))/len(rms)
        initial.append(init)
        details.append({"pde":spec.name,"residuals":integrals,"scales":scales,"initial_difference":initial_difference,
                        "physical_weak_MSE":[(r.square().mean(-1)*tw).sum() for r in integrals],
                        "physical_initial_MSE":[(d.square()*rule.spatial_weights).sum() for d in initial_difference]})
    return PhysicalLosses(torch.stack(weak),torch.stack(initial),details,timing)


def strong_diagnostics(model,context,queries):
    """Independent diagnostic points; classical residuals, never a default loss.

    Normalization uses training-prior strong-term RMS: source/speed/time-term
    scales in L2 space. Wave equations have distinct physical residual units.
    """
    q=QueryBatch(queries.position.clone().requires_grad_(),queries.physical_time.clone().requires_grad_(),queries.time_present,queries.quantity_id,queries.valid)
    derivative=field_derivatives(model,context,q,create_graph=False,second=True)
    xi=model.infer_unknowns(context).xi_hat
    result=[]
    # Arbitrary-coordinate modal source uses the existing basis constants, copied
    # explicitly; this diagnostic has no parameter-gradient requirement.
    for row,spec in enumerate(context.specs):
        valid=q.valid[row];lam=torch.tensor([v for v in eigenvalues().tolist()],device=q.position.device,dtype=q.position.dtype)
        tau=q.position.new_tensor(spec.parameter_blocks[0].prior.scale)
        if spec.name=="poisson":
            # Explicit two-step constant evaluation: MPS cannot even perform a
            # combined transfer/cast to Double. No differentiated neural value
            # or derivative crosses devices here; only diagnostic basis values.
            position_cpu=q.position[row,valid].detach().to(device="cpu").to(dtype=torch.float64)
            phi=basis(position_cpu).to(dtype=q.position.dtype).to(device=q.position.device)
            source=(xi[row,:9]*lam)@phi.T
            residuals=[-derivative.laplacian_u[row,valid,0]-source];scales=[(lam*tau).square().sum().sqrt()];units=["U/L^2"]
        elif spec.name=="heat":
            alpha=known_scalar(spec,"alpha");residuals=[derivative.u_t[row,valid,0]-alpha*derivative.laplacian_u[row,valid,0]];scales=[(alpha*lam*tau).square().sum().sqrt()];units=["U/T"]
        else:
            c=known_scalar(spec,"c");omega=c*lam.sqrt()
            residuals=[derivative.u_t[row,valid,0]-derivative.v[row,valid,0],derivative.v_t[row,valid,0]-c*c*derivative.laplacian_u[row,valid,0]]
            scales=[(omega*tau).square().sum().sqrt(),(omega.square()*tau).square().sum().sqrt()];units=["U/T","U/T^2"]
        result.append({"pde":spec.name,"units":units,"residual_values":[v.detach() for v in residuals],
                       "physical_RMSE":[float(v.detach().square().mean().sqrt()) for v in residuals],
                       "normalized_RMSE":[float((v.detach()/s).square().mean().sqrt()) for v,s in zip(residuals,scales)]})
    return result

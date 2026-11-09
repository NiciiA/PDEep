"""Explicit CPU-float64 physical bridge for B; reuses validated T1 operators.

Neural tensors remain float32 on the selected device. Transfer/cast operations
retain autograd (no detach). MPS transfers to CPU BEFORE any float64 cast. The
same sensor_matrix/evaluate_batch used by reference/D supply all physical values.
"""
from dataclasses import dataclass,fields
import time
import torch
from .batches import UnknownCoefficients,expected_active_mask,masked_values
from .bayes import sensor_matrix
from .reference import require_supported,evaluate_batch,basis,eigenvalues
from .core import require


def cpu64(t):return t.to(device='cpu').to(dtype=torch.float64)


def physical_batch(batch):
    return type(batch)(**{f.name:cpu64(getattr(batch,f.name)) if getattr(batch,f.name).is_floating_point()
                          else getattr(batch,f.name).to('cpu') for f in fields(batch)})


@dataclass(frozen=True)
class PhysicalSystem:
    specs: tuple
    H: torch.Tensor # [B,N,18], CPU64
    y: torch.Tensor
    sigma: torch.Tensor
    valid: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    active: torch.Tensor
    curvature: torch.Tensor # largest prior-whitened precision eigenvalue
    construction_seconds: float

    def state(self,xi):
        require(xi.shape==self.mean.shape and xi.dtype==torch.float64 and xi.device.type=='cpu','B physical xi must be CPU64 [B,18]')
        prediction=(self.H@xi[...,None])[...,0]
        residual=masked_values(prediction-self.y,self.valid)
        weighted=residual/self.sigma.square()
        gradient=(self.H.transpose(1,2)@weighted[...,None])[...,0]+masked_values((xi-self.mean)/self.scale.square(),self.active)
        objective=.5*((residual/self.sigma).square().sum(1)+masked_values(((xi-self.mean)/self.scale).square(),self.active).sum(1))
        require(bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(objective).all()),'nonfinite B physical objective')
        return prediction,residual,gradient,objective


def build_system(specs,observations):
    start=time.perf_counter();obs=physical_batch(observations).validate(specs);clean=obs.masked();B,N=clean.valid.shape
    rows=[];means=[];scales=[]
    for i,s in enumerate(specs):
        require_supported(s)
        matrix=sensor_matrix(s,obs,row=i)
        rows.append(torch.nn.functional.pad(matrix,(0,18-matrix.shape[-1])))
        m=[x for block in s.parameter_blocks for x in block.prior.mean];sd=[x for block in s.parameter_blocks for x in block.prior.scale]
        means.append(m+[0.]*(18-len(m)));scales.append(sd+[1.]*(18-len(sd)))
    H=torch.stack(rows);mean=H.new_tensor(means);scale=H.new_tensor(scales);active=expected_active_mask(specs)
    sigma=torch.where(clean.valid,clean.sigma[...,0],torch.ones_like(clean.sigma[...,0]))
    A=H*scale[:,None]/sigma[...,None]
    K=torch.eye(18,dtype=torch.float64)[None]+A.transpose(1,2)@A
    curvature=torch.linalg.eigvalsh(K)[:,-1]
    return PhysicalSystem(tuple(specs),H,clean.value[...,0],sigma,clean.valid,mean,scale,active,curvature,time.perf_counter()-start)


def physical_solution(system,xi,queries):
    q=physical_batch(queries)
    return evaluate_batch(system.specs,UnknownCoefficients(xi,system.active),q)


def anchor_projections(system,xi,gradient,positions):
    """Candidate fields and pushforward of EUCLIDEAN coefficient gradient.

    Poisson xi=a parameterizes solution coefficients; source amplitude is lambda*a.
    We display/project lambda*g_a into source space. This is NOT the covector
    gradient wrt source amplitudes (that would be g_a/lambda), nor a functional
    gradient in an unspecified field metric. Raw and asinh-normalized projections
    are separately exposed. Dynamic blocks are initial u0 / v0.
    """
    phi=basis(cpu64(positions));lam=eigenvalues();fields=[];gradients=[];features=[]
    for i,s in enumerate(system.specs):
        transform=torch.cat((lam,torch.zeros_like(lam))) if s.name=='poisson' else torch.cat((torch.ones_like(lam),torch.ones_like(lam) if s.name=='wave' else torch.zeros_like(lam)))
        x=xi[i].reshape(2,9)*transform.reshape(2,9)
        g=gradient[i].reshape(2,9)*transform.reshape(2,9)
        values=phi@x.T;projected=phi@g.T
        rms=(system.scale[i].reshape(2,9)*transform.reshape(2,9)).square().sum(1).sqrt().clamp_min(1e-12)
        # Prior-preconditioned direction divided by curvature has physical field
        # units. Raw gradient projection above is retained for inspection.
        direction=phi@((gradient[i]*system.scale[i].square()/system.curvature[i]).reshape(2,9)*transform.reshape(2,9)).T
        present=values.new_tensor([1.,float(s.name=='wave')]).expand(len(phi),-1)
        features.append(torch.cat((values/rms,torch.asinh(projected/rms),direction/rms,present),-1))
        fields.append(values);gradients.append(projected)
    return torch.stack(fields),torch.stack(gradients),torch.stack(features)


def fixed_refinement(system,initial=None,steps=3):
    """Nonlearned diagonal-prior preconditioner, eta=1/lambda_max(K)."""
    xi=system.mean.clone() if initial is None else initial
    estimates=[xi];objectives=[system.state(xi)[-1]]
    for _ in range(steps):
        g=system.state(xi)[2];xi=masked_values(xi-system.scale.square()*g/system.curvature[:,None],system.active)
        estimates.append(xi);objectives.append(system.state(xi)[-1])
    return torch.stack(estimates,1),torch.stack(objectives,1)

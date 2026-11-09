"""Bounded independent mathematical checks, shared by tests and the notebook."""
from __future__ import annotations

import math
import time
import numpy as np
import torch

from .core import require
from .data import CaseDataset
from .diagnostics import RandomStreams
from .reference import basis, basis_gradient, basis_laplacian, eigenvalues, evaluate


ATOL = 1e-12
RTOL = 1e-10


def gradient(value, coordinate):
    return torch.autograd.grad(value.sum(), coordinate, create_graph=True)[0]


def laplacian_ad(value, position):
    first = gradient(value, position)
    return sum(gradient(first[:, j:j+1], position)[:, j:j+1] for j in range(2))


def numpy_oracle(name, xi, position, physical_time=None, quantity="u"):
    """Independent scalar mode loop; no shared basis/rate/derivative helpers."""
    result = np.zeros(len(position))
    for j, (m, n) in enumerate((m,n) for m in range(1,4) for n in range(1,4)):
        phi = 2*np.sin(m*np.pi*position[:,0])*np.sin(n*np.pi*position[:,1])
        lam = np.pi**2 * (m*m+n*n)
        a = xi[j]
        if name == "poisson":
            amplitude = a
        elif name == "heat":
            amplitude = a*np.exp(-.02*lam*physical_time[:,0])
        else:
            omega = .7*np.sqrt(lam)
            b = xi[j+9]
            t = physical_time[:,0]
            amplitude = (a*np.cos(omega*t)+b/omega*np.sin(omega*t)) if quantity=="u" else (-omega*a*np.sin(omega*t)+b*np.cos(omega*t))
        result += amplitude*phi
    return result[:,None]


def mathematical_checks(root_seed=20261108, *, points=31, cases_per_pde=3):
    require(points > 0 and cases_per_pde > 0, "positive verification budget required")
    started = time.perf_counter()
    rng = RandomStreams(root_seed).numpy("tests", "reference-mathematics")
    p = torch.from_numpy(rng.uniform(.01,.99,(points,2))).requires_grad_()
    times = torch.from_numpy(rng.uniform(.0001,.9999,(points,1))).requires_grad_()
    checks = []
    def record(name, residual, *terms, boundary=False):
        error = float(residual.detach().abs().max())
        scale = max((float(t.detach().abs().max()) for t in terms), default=1.)
        limit = ATOL if boundary else ATOL+RTOL*scale
        checks.append({"name":name, "max_abs_error":error, "physical_term_scale":scale,
                       "scaled_error":error/max(scale,1.), "tolerance":limit, "passed":error <= limit})
    phi = basis(p)
    record("basis_eigenvalue_identity", -basis_laplacian(p)-phi*eigenvalues(), phi*eigenvalues())
    grads, laps = [], []
    for j in range(9):
        grads.append(gradient(phi[:,j:j+1],p))
        laps.append(laplacian_ad(phi[:,j:j+1],p))
    record("basis_gradient_vs_autograd", basis_gradient(p)-torch.stack(grads,dim=1), torch.stack(grads,dim=1))
    record("basis_laplacian_vs_autograd", basis_laplacian(p)-torch.cat(laps,dim=1), torch.cat(laps,dim=1))
    edge = torch.linspace(0.,1.,points,dtype=torch.float64)
    boundary = torch.cat((torch.stack((edge,edge*0),-1),torch.stack((edge,edge*0+1),-1),
                          torch.stack((edge*0,edge),-1),torch.stack((edge*0+1,edge),-1)))
    dataset = CaseDataset(root_seed)
    for name in ("poisson","heat","wave"):
        for case_id in range(cases_per_pde):
            xi = dataset.case("train",name,case_id).tensor()
            t = None if name == "poisson" else times
            u = evaluate(name,xi,p,t)
            lap = laplacian_ad(u,p)
            prefix=f"{name}/case{case_id}"
            oracle = torch.from_numpy(numpy_oracle(name,xi.numpy(),p.detach().numpy(), None if t is None else t.detach().numpy()))
            record(prefix+"/continuous_numpy_oracle",u-oracle,u)
            analytic_lap = evaluate(name,xi,p,t,spatial_derivative=(2,0))+evaluate(name,xi,p,t,spatial_derivative=(0,2))
            record(prefix+"/spatial_second_derivatives",analytic_lap-lap,lap)
            bt = None if name == "poisson" else torch.linspace(0.,1.,len(boundary),dtype=torch.float64)[:,None]
            record(prefix+"/boundary_u",evaluate(name,xi,boundary,bt),boundary=True)
            if name == "poisson":
                f = evaluate(name,xi,p,quantity="f")
                record(prefix+"/poisson_pde",-lap-f,lap,f)
            else:
                ut = gradient(u,times)
                record(prefix+"/analytic_dt_vs_autograd",evaluate(name,xi,p,t,time_order=1)-ut,ut)
                initial = evaluate(name,xi,p,torch.zeros_like(times))
                u0 = evaluate(name,xi,p,quantity="u0")
                record(prefix+"/initial_u",initial-u0,u0)
                if name == "heat":
                    record(prefix+"/heat_pde",ut-.02*lap,ut,.02*lap)
                else:
                    v = evaluate(name,xi,p,t,quantity="v")
                    utt = gradient(ut,times)
                    vt = gradient(v,times)
                    record(prefix+"/wave_pde",utt-.7**2*lap,utt,.7**2*lap)
                    record(prefix+"/wave_first_order",vt-.7**2*lap,vt,.7**2*lap)
                    record(prefix+"/velocity_equals_dt_u",v-ut,v,ut)
                    record(prefix+"/analytic_dtt_vs_autograd",evaluate(name,xi,p,t,time_order=2)-utt,utt)
                    voracle = torch.from_numpy(numpy_oracle(name,xi.numpy(),p.detach().numpy(),t.detach().numpy(),"v"))
                    record(prefix+"/velocity_numpy_oracle",v-voracle,v)
                    v0 = evaluate(name,xi,p,quantity="v0")
                    record(prefix+"/initial_v",evaluate(name,xi,p,torch.zeros_like(times),quantity="v")-v0,v0)
                    record(prefix+"/boundary_v",evaluate(name,xi,boundary,bt,quantity="v"),boundary=True)
    return {"device":"cpu", "dtype":"float64", "points":points, "cases_per_pde":cases_per_pde,
            "rtol":RTOL,"atol":ATOL,"temporal_interpolation":"none; direct continuous evaluation",
            "elapsed_seconds":time.perf_counter()-started, "passed":all(c["passed"] for c in checks),
            "max_abs_error":max(c["max_abs_error"] for c in checks), "checks":checks}

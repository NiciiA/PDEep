"""C-Geo domain weak integrals with FE test functions and autograd coordinates.

Each full residual is accumulated over all quadrature chunks BEFORE squaring.
Only first weak spatial/time derivatives are used. Unknown Poisson source comes
from the model's auxiliary head, never from hidden case targets. BC G1 = known
homogeneous Dirichlet; general FEM mixed BCs are separate verified references.
"""
from dataclasses import dataclass
import numpy as np
import torch
from ..core import require
from ..train_types import LossReport
from .quadrature import cells,boundary
from .representation import tensor


@dataclass(frozen=True)
class WeakConfig:
    spatial_order: int=3
    time_order: int=3
    tests: int=16
    chunk: int=256
    weak_weight: float=.1
    initial_weight: float=.1
    boundary_weight: float=.1
    auxiliary_weight: float=.1


def derivative(y,x):
    return torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]


def weak_loss(model,contexts,normalizers,config=WeakConfig()):
    values=[];boundary_losses=[];initial_losses=[];details=[]
    for context in contexts:
        p=context.problem;mesh=p.mesh;like=context.equation;q=cells(mesh,config.spatial_order)
        conditions=p.conditions;dir_tags={k for k,b in conditions.items() if b.kind=='DIRICHLET'}
        dir_nodes=np.unique(mesh.boundary_edges[np.isin(mesh.boundary_tags,list(dir_tags))])
        interior=np.setdiff1d(np.arange(len(mesh.vertices)),dir_nodes);coords=mesh.vertices[interior];order=np.lexsort((coords[:,1],coords[:,0]));tests=interior[order[np.linspace(0,len(interior)-1,min(config.tests,len(interior)),dtype=int)]]
        require(len(tests)>0,'no admissible zero-trace FE tests')
        # FE hat values/gradients at quadrature points. Gradients are independently
        # assembled here from the geometric affine map, not the FEM matrix.
        tri=mesh.vertices[mesh.cells];J=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
        grad=np.array([[-1.,-1.],[1.,0.],[0.,1.]])[None]@np.linalg.inv(J)
        psi=np.zeros((len(q.position),len(tests)));dpsi=np.zeros((len(q.position),len(tests),2))
        for j,node in enumerate(tests):
            hit=mesh.cells[q.cell_id]==node;psi[:,j]=(q.barycentric*hit).sum(1);dpsi[:,j]=(grad[q.cell_id]*hit[...,None]).sum(1)
        W=tensor(q.weight,like);P=tensor(psi,like);DP=tensor(dpsi,like)
        qb=boundary(mesh,order=config.spatial_order);bi,bw=mesh.locate(qb.position);BP=np.zeros((len(qb.position),len(tests)))
        for j,node in enumerate(tests):BP[:,j]=(bw*(mesh.cells[bi]==node)).sum(1)
        flux=np.array([conditions[int(k)].value if conditions[int(k)].kind=='NEUMANN' else 0. for k in qb.boundary_tag])
        natural=tensor((qb.weight[:,None]*flux[:,None]*BP).sum(0),like)
        L=np.sqrt(mesh.cell_areas.sum());su=normalizers.scale(p.name,'u');sv=normalizers.scale(p.name,'v') if p.name=='wave' else 0.
        area=(W[:,None]*P.abs()).sum(0);gradient=(W[:,None]*DP.norm(dim=-1)).sum(0)
        # Fixed physical-term units and training-only quantity RMS; no residual-
        # adaptive scaling, vertex-count normalization or test-data calibration.
        scale=(su*gradient/L+(normalizers.scale(p.name,'f')*area if p.name=='poisson' else su*area+p.coefficient*su*gradient/L))
        if p.name=='wave':scale=sv*area+p.coefficient*su*gradient/L
        require(bool((scale>0).all()),'zero weak test normalization')
        if p.physical_time is None:times=[0.];weights=[1.]
        else:z,w=np.polynomial.legendre.leggauss(config.time_order);times=(z+1)/2;weights=w/2
        per_time=[];rmax=0.
        for t,tw in zip(times,weights):
            residual=-natural;kinematic=like.new_zeros(len(tests))
            for a in range(0,len(W),config.chunk):
                s=slice(a,min(a+config.chunk,len(W)));x=tensor(q.position[s],like).requires_grad_(True);time=torch.full((len(x),1),float(t),device=x.device,dtype=x.dtype,requires_grad=True)
                u=model.evaluate(context,x,None if p.physical_time is None else time,'u');du=derivative(u,x)
                stiffness=(W[s,None]*p.coefficient*(du[:,None,:]*DP[s]).sum(-1)).sum(0)
                if p.name=='poisson':
                    source=model.evaluate(context,x,None,'f')[:,0];residual=residual+stiffness-(W[s,None]*source[:,None]*P[s]).sum(0)
                elif p.name=='heat':
                    ut=derivative(u,time)[:,0];residual=residual+stiffness+(W[s,None]*ut[:,None]*P[s]).sum(0)
                else:
                    v=model.evaluate(context,x,time,'v');vt=derivative(v,time)[:,0];ut=derivative(u,time)[:,0]
                    residual=residual+stiffness+(W[s,None]*vt[:,None]*P[s]).sum(0)
                    kinematic=kinematic+(W[s,None]*(ut-v[:,0])[:,None]*P[s]).sum(0)
            term=(residual/scale).square().mean()
            if p.name=='wave':term=.5*(term+(kinematic/((su+sv)*area)).square().mean())
            per_time.append(float(tw)*term);rmax=max(rmax,float(residual.detach().abs().max()))
        values.append(sum(per_time))
        # Boundary losses use tagged boundary quadrature and physical measure.
        x=tensor(qb.position,like).requires_grad_(True);wb=tensor(qb.weight,like);bc_terms=[]
        for t,tw in zip(times,weights):
            time=x.new_full((len(x),1),float(t)) if p.physical_time is not None else None
            u=model.evaluate(context,x,time,'u');du=derivative(u,x);normal=tensor(qb.normal,x);pieces=[]
            for tag,bc in conditions.items():
                take=torch.as_tensor(qb.boundary_tag==tag,device=x.device);weight=wb[take]
                if bc.kind=='DIRICHLET':
                    e=(u[take,0]-bc.value)/su;term=(weight*e.square()).sum()
                    if p.name=='wave':
                        v=model.evaluate(context,x,time,'v');term=.5*(term+(weight*(v[take,0]/sv).square()).sum())
                else:
                    e=(p.coefficient*(du[take]*normal[take]).sum(-1)-bc.value)/(p.coefficient*su/L+abs(bc.value));term=(weight*e.square()).sum()
                pieces.append(term)
            bc_terms.append(float(tw)*sum(pieces)/wb.sum())
        if not dir_tags and p.name=='poisson':
            xg=tensor(q.position,like);mean=(W*model.evaluate(context,xg,None,'u')[:,0]).sum()/W.sum()
            bc_terms.append((mean/su).square())  # explicit zero-mean Neumann gauge
        boundary_losses.append(sum(bc_terms))
        if p.name!='poisson':
            x=tensor(q.position,like);t=x.new_zeros((len(x),1));terms=[]
            for state,initial in (('u','u0'),('v','v0')) if p.name=='wave' else (('u','u0'),):
                diff=(model.evaluate(context,x,t,state)-model.evaluate(context,x,t,initial))[:,0]/normalizers.scale(p.name,initial)
                terms.append((W*diff.square()).sum()/W.sum())
            initial_losses.append(sum(terms)/len(terms))
        else:initial_losses.append(like.sum()*0)
        details.append({'quadrature_points':len(W),'tests':len(tests),'time_nodes':len(times),'max_raw_integrated_residual':rmax,
                        'derivatives':'first weak; element interiors; C0/H1 geometry path','boundary':'tagged known constant D/N; Neumann natural load; pure-N zero-mean gauge'})
    return {'weak':torch.stack(values).mean(),'boundary':torch.stack(boundary_losses).mean(),'initial':torch.stack(initial_losses).mean(),'diagnostics':details}


def supervised(model,batch,contexts):
    prediction=model.predict(contexts,batch.queries);field=[];aux=[]
    for row,(context,case) in enumerate(zip(contexts,batch.cases)):
        valid=batch.queries.valid[row];q=batch.queries.quantity_id[row];terms=[]
        for quantity in (0,1) if case.problem.name=='wave' else (0,):
            take=valid&(q==quantity)
            if take.any():terms.append(((prediction[row,take]-batch.queries.value[row,take])/batch.normalizers.scale(case.problem.name,quantity)).square().mean())
        require(bool(terms),'each training case requires valid field targets');field.append(sum(terms)/len(terms))
        x=batch.queries.position[row,valid];t=x.new_zeros((len(x),1)) if case.problem.physical_time is not None else None;terms=[]
        for name,hidden in case.fields.items():
            # Explicit hidden target supervision, never an input to prepare_context.
            truth=tensor(hidden(x.detach().cpu().double().numpy()),x)[:,None]
            pred=model.evaluate(context,x,t,name);terms.append(((pred-truth)/batch.normalizers.scale(case.problem.name,name)).square().mean())
        aux.append(sum(terms)/len(terms))
    return torch.stack(field).mean(),torch.stack(aux).mean()


class Adapter:
    def __init__(self,architecture,weak=WeakConfig()):self.architecture=architecture;self.weak=weak
    def loss(self,model,batch,*,detailed=False):
        contexts=model.prepare_context(batch.specs,batch.observations);f,a=supervised(model,batch,contexts)
        raw={'field':f,'auxiliary':a};weighted={'field':f,'auxiliary':self.weak.auxiliary_weight*a};diagnostics={}
        if self.architecture=='C_GEO' and model.config.physics:
            physical=weak_loss(model,contexts,batch.normalizers,self.weak);diagnostics=physical.pop('diagnostics');raw.update(physical)
            weighted.update(weak=self.weak.weak_weight*raw['weak'],boundary=self.weak.boundary_weight*raw['boundary'],initial=self.weak.initial_weight*raw['initial'])
        counts={'cases':len(batch.cases),'valid_queries':int(batch.queries.valid.sum()),'active_coefficients':0,'valid_observations':int(batch.observations.valid.sum())}
        meanings={k:'G1 equal-case, quantity-balanced; physical RMS standardized' for k in raw}
        return LossReport(sum(weighted.values()),raw,weighted,meanings,counts,{'weak':diagnostics} if detailed else {})

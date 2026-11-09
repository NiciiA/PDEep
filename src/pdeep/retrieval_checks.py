"""Bounded smooth-retrieval audit; numerical probes only, never training."""
from dataclasses import fields,replace
import math
import time
import numpy as np
import torch

from .batches import QueryBatch
from .core import require
from .data import CaseDataset,calibrate,observations,collate_inputs
from .encoding import EquationVocabulary,pack_graphs
from .geometry import geometry_for
from .memory import gaussian_readout
from .reference import canonical_spec
from .representations import build_representations,float32_batch,parameter_inventory
from .runtime import resolve_device


def gaussian_oracle(position,anchors,tokens,ell):
    """Independent NumPy ratio-of-exponentials plus closed-form first derivative."""
    delta=position[:,:,None,:]-anchors[None,None,:,:]
    unnormalized=np.exp(-(delta**2).sum(-1)/(2*ell**2))
    weights=unnormalized/unnormalized.sum(-1,keepdims=True)
    mean_anchor=weights@anchors
    dw=weights[...,None]*(anchors[None,None]-mean_anchor[:,:,None])/ell**2
    return weights@tokens,weights,np.einsum('bqmd,bmf->bqfd',dw,tokens)


def kernel_checks():
    layout=geometry_for().anchor_layout()
    probes=torch.tensor([[(x,y) for x in (.125,.25,.5,.75,.875) for y in (.37,.5,.63)]],dtype=torch.float64)
    checks=[];scales=[]
    def record(name,error,tolerance):
        error=float(error.detach()) if isinstance(error,torch.Tensor) else float(error)
        checks.append({"name":name,"max_abs_error":error,"tolerance":tolerance,"passed":error<=tolerance})
    for level,ell in ((0,.30),(1,.15)):
        anchors=layout.positions[layout.levels==level].double()
        tokens=torch.stack(((13*anchors[:,0]).sin()+(7*anchors[:,1]).cos(),anchors[:,0]+2*anchors[:,1],
                             (3*anchors.sum(-1)).sin()),dim=-1)[None]
        p=probes.clone().requires_grad_()
        value,weights=gaussian_readout(p,anchors,tokens,ell)
        expected,ew,derivative=gaussian_oracle(p.detach().numpy(),anchors.numpy(),tokens.numpy(),ell)
        auto=torch.stack([torch.autograd.grad(value[...,i].sum(),p,retain_graph=True)[0] for i in range(3)],dim=-2)
        record(f"scale{level}/formula",(value-torch.from_numpy(expected)).abs().max(),2e-14)
        record(f"scale{level}/weights",(weights-torch.from_numpy(ew)).abs().max(),2e-14)
        record(f"scale{level}/analytic_derivative",(auto-torch.from_numpy(derivative)).abs().max(),2e-13)
        h=1e-5
        differences=[]
        for axis in range(2):
            offset=torch.zeros_like(p);offset[...,axis]=h
            differences.append((gaussian_readout(p.detach()+offset,anchors,tokens,ell)[0]-
                                gaussian_readout(p.detach()-offset,anchors,tokens,ell)[0])/(2*h))
        finite=torch.stack(differences,dim=-1)
        record(f"scale{level}/finite_difference_derivative",(auto-finite).abs().max(),1e-7)
        record(f"scale{level}/normalization",(weights.sum(-1)-1).abs().max(),1e-14)
        require(bool((weights>0).all()),"default Gaussian must include every anchor")
        sequence=[]
        for eps in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-7):
            offset=torch.zeros_like(probes);offset[...,0]=eps
            left=(probes-offset).requires_grad_();right=(probes+offset).requires_grad_()
            lv,_=gaussian_readout(left,anchors,tokens,ell);rv,_=gaussian_readout(right,anchors,tokens,ell)
            ld=torch.stack([torch.autograd.grad(lv[...,i].sum(),left,retain_graph=True)[0] for i in range(3)],dim=-2)
            rd=torch.stack([torch.autograd.grad(rv[...,i].sum(),right,retain_graph=True)[0] for i in range(3)],dim=-2)
            sequence.append({"epsilon":eps,"value_difference":float((rv-lv).detach().abs().max()),
                             "derivative_difference":float((rd-ld).detach().abs().max())})
        ratios=[sequence[i+1]["value_difference"]/sequence[i]["value_difference"] for i in range(5)]
        require(max(ratios)<.11,"Gaussian value continuity did not converge linearly")
        derivative_ratios=[sequence[i+1]["derivative_difference"]/sequence[i]["derivative_difference"] for i in range(5)]
        require(max(derivative_ratios)<.11,"Gaussian derivative continuity did not converge linearly")
        # gradcheck compares all outputs/coordinates on a small transition-crossing set.
        small=probes[:,:3].clone().requires_grad_()
        passed=torch.autograd.gradcheck(lambda x:gaussian_readout(x,anchors,tokens,ell)[0],(small,),eps=1e-6,atol=1e-7,rtol=1e-5)
        require(passed,"Gaussian gradcheck failed")
        scales.append({"level":level,"anchors":len(anchors),"lengthscale":ell,"transition_probes":len(probes[0]),
                       "continuity":sequence,"value_step_ratios":ratios,"derivative_step_ratios":derivative_ratios,"gradcheck":passed})
    return {"passed":all(c["passed"] for c in checks),"dtype":"float64","device":"cpu","checks":checks,"scales":scales}


def query_at_x(x,specs,device):
    x=torch.as_tensor(x,dtype=torch.float32,device=device)
    p=torch.stack((x,torch.full_like(x,.5)),dim=-1)[None].repeat(len(specs),1,1)
    present=torch.tensor([s.physical_time is not None for s in specs],device=device)[:,None].expand(-1,len(x))
    t=torch.where(present[...,None],torch.full((*present.shape,1),.2,device=device),torch.zeros((*present.shape,1),device=device))
    return QueryBatch(p,t,present,torch.zeros_like(present,dtype=torch.int64),torch.ones_like(present))


def retrieval_checks(*,device="cpu",root_seed=20261108):
    resolved,_=resolve_device(device,"float32")
    started=time.perf_counter()
    specs=tuple(canonical_spec(p) for p in ("poisson","heat","wave"))
    norm=calibrate(root_seed);dataset=CaseDataset(root_seed)
    vocab=EquationVocabulary.from_specs(specs)
    a=build_representations("A",norm,vocab,root_seed=root_seed).to(resolved).eval()
    obs=float32_batch(collate_inputs([observations(dataset,dataset.case("train",s.name,0),norm).inputs for s in specs]),device=device)
    with torch.no_grad():
        eq=a.equation(pack_graphs(specs,vocab).to(resolved));encoded=a.observation(obs,specs)
        state=a.memory(encoded,eq.context)
    points=[.125-1e-4,.125,.125+1e-4,.25-1e-4,.25,.25+1e-4,.5-1e-4,.5,.5+1e-4,.75,.875]
    q=query_at_x(points,specs,resolved)
    p=q.position.detach().requires_grad_();q=replace(q,position=p)
    def retrieve(query):
        feature=a.query(query,specs,eq.context,encoded.summary)
        return a.retrieval(state,query,feature,specs)
    out=retrieve(q)
    channels=[0,17,95]
    derivative=torch.stack([torch.autograd.grad(out.features[...,c].sum(),p,retain_graph=True)[0] for c in channels],dim=-2)
    require(bool(torch.isfinite(derivative).all()) and float(derivative.abs().sum())>0,"nonfinite/missing fused query derivatives")
    h=2e-4
    with torch.no_grad():
        fd=[]
        for axis in range(2):
            offset=torch.zeros_like(p);offset[...,axis]=h
            fd.append((retrieve(replace(q,position=p+offset)).features[...,channels]-
                       retrieve(replace(q,position=p-offset)).features[...,channels])/(2*h))
        finite=torch.stack(fd,dim=-1)
        absolute=float((finite-derivative).abs().max())
        scaled=float(((finite-derivative).abs()/(1+derivative.abs())).max())
        require(scaled<3e-3,"float32 fused derivative finite-difference mismatch")
        split=[];individual=[]
        for lo,hi in ((0,3),(3,7),(7,len(points))):
            chunk=QueryBatch(*(getattr(q,k)[:,lo:hi] for k in ("position","physical_time","time_present","quantity_id","valid")))
            split.append(retrieve(chunk).features)
        for i in range(len(points)):
            single=QueryBatch(*(getattr(q,k)[:,i:i+1] for k in ("position","physical_time","time_present","quantity_id","valid")))
            individual.append(retrieve(single).features)
        chunk_error=float((torch.cat(split,1)-out.features).abs().max())
        individual_error=float((torch.cat(individual,1)-out.features).abs().max())
        require(max(chunk_error,individual_error)<1e-5,"query chunk/batched-individual disagreement")
        require(out.weights.shape[-1]==80 and bool((out.weights>0).all()),"V1 retrieval must include all 80 anchors")
        weight_sum_error=max(float((out.weights[:,:,:16].sum(-1)-1).abs().max()),float((out.weights[:,:,16:].sum(-1)-1).abs().max()))
        require(weight_sum_error<1e-6,"scale weights not separately normalized")
        sequence=[]
        for epsilon in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-7):
            crossing=query_at_x([.5-epsilon,.5+epsilon],specs,resolved)
            result=retrieve(crossing)
            sequence.append({"epsilon":epsilon,"feature_difference_per_pde":(result.features[:,1]-result.features[:,0]).abs().amax(1).cpu().tolist()})
        require(max(sequence[-1]["feature_difference_per_pde"])<1e-4,"fused retrieval has a finite transition jump")
    if resolved.type=="mps":torch.mps.synchronize()
    if resolved.type=="cuda":torch.cuda.synchronize(resolved)
    return {"passed":True,"device":str(resolved),"dtype":"float32","elapsed_seconds":time.perf_counter()-started,
            "parameters":parameter_inventory(a),"output_shape":list(out.features.shape),"weights_shape":list(out.weights.shape),
            "coarse_lengthscale":float(a.retrieval.lengthscales[0]),"fine_lengthscale":float(a.retrieval.lengthscales[1]),
            "transition_points":points,"first_derivative_abs_max":float(derivative.abs().max()),
            "derivative_probe_channels":channels,
            "finite_difference_step":h,"finite_difference_max_abs_error":absolute,
            "finite_difference_max_scaled_error":scaled,"finite_difference_scaled_tolerance":3e-3,
            "query_chunk_max_error":chunk_error,"batched_individual_max_error":individual_error,
            "weight_sum_max_error":weight_sum_error,"transition_continuity":sequence}

"""Bounded feature diagnostics, not a scientific prediction/training experiment."""
from dataclasses import fields,replace
import time
import torch

from .batches import QueryBatch
from .core import require
from .data import CaseDataset,calibrate,collate_inputs,observations,training_queries
from .encoding import EquationVocabulary,CoordinateFeatures,pack_graphs
from .memory import coefficient_inputs
from .reference import canonical_spec
from .representations import build_representations,float32_batch,parameter_inventory,computational_estimate
from .runtime import resolve_device


def padded(batch,count):
    result={}
    for field in fields(batch):
        tensor=getattr(batch,field.name)
        fill=False if tensor.dtype==torch.bool else (-999 if tensor.dtype==torch.int64 else float("nan"))
        extra=torch.full((len(tensor),count,*tensor.shape[2:]),fill,device=tensor.device,dtype=tensor.dtype)
        result[field.name]=torch.cat((tensor,extra),dim=1)
    return type(batch)(**result)


def stats(tensor,valid=None):
    value=(tensor if valid is None else tensor[valid]).detach()
    return {"shape":list(tensor.shape),"device":str(tensor.device),"dtype":str(tensor.dtype),
            "finite":bool(torch.isfinite(value).all()),"mean":float(value.mean()),
            "std":float(value.std(unbiased=False)),"max_abs":float(value.abs().max())}


def sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)
    elif device.type=="mps":torch.mps.synchronize()


def representation_checks(*,device="cpu",root_seed=20261108,normalizers=None):
    resolved,_=resolve_device(device,"float32")
    began=time.perf_counter()
    specs=tuple(canonical_spec(p) for p in ("poisson","heat","wave"))
    normalizers=calibrate(root_seed) if normalizers is None else normalizers
    dataset=CaseDataset(root_seed)
    cases=tuple(dataset.case("train",s.name,0) for s in specs)
    obs=float32_batch(collate_inputs([observations(dataset,c,normalizers).inputs for c in cases]),device=device)
    query=float32_batch(collate_inputs([training_queries(dataset,c).inputs for c in cases]),device=device)
    vocabulary=EquationVocabulary.from_specs(specs)
    rng_before=torch.random.get_rng_state().clone()
    owners={owner:build_representations(owner,normalizers,vocabulary,root_seed=root_seed) for owner in ("A","B","C")}
    require(torch.equal(rng_before,torch.random.get_rng_state()),"representation construction changed global RNG")
    pointers=[{p.data_ptr() for p in m.parameters()} for m in owners.values()]
    require(not(pointers[0]&pointers[1] or pointers[0]&pointers[2] or pointers[1]&pointers[2]),"cross-owner parameter sharing")
    inventories={owner:parameter_inventory(module) for owner,module in owners.items()}
    a=owners["A"].to(resolved).eval()
    graph=pack_graphs(specs,vocabulary).to(resolved)
    checks=[]
    def record(name,error,tolerance=1e-5):
        error=float(error.detach()) if isinstance(error,torch.Tensor) else float(error)
        checks.append({"name":name,"max_abs_error":error,"tolerance":tolerance,"passed":error<=tolerance})
    with torch.no_grad():
        eq=a.equation(graph)
        encoded=a.observation(obs,specs)
        state=a.memory(encoded,eq.context,return_attention=True)
        q_features=a.query(query,specs,eq.context,encoded.summary)
        retrieved=a.retrieval(state,query,q_features,specs)
        padded_obs=a.observation(padded(obs,11),specs)
        record("observation_padding_tokens",(padded_obs.tokens[:,:64]-encoded.tokens).abs().max())
        record("observation_padding_summary",(padded_obs.summary-encoded.summary).abs().max())
        state_pad=a.memory(padded_obs,eq.context)
        record("memory_padding",(state_pad.tokens-state.tokens).abs().max())
        order=torch.arange(63,-1,-1,device=resolved)
        shuffled=type(obs)(**{f.name:getattr(obs,f.name)[:,order] for f in fields(obs)})
        perm=a.observation(shuffled,specs)
        record("observation_permutation_tokens",(perm.tokens-encoded.tokens[:,order]).abs().max())
        record("observation_permutation_summary",(perm.summary-encoded.summary).abs().max())
        state_perm=a.memory(perm,eq.context,return_attention=True)
        record("memory_permutation",(state_perm.tokens-state.tokens).abs().max())
        record("attention_permutation",(state_perm.attention-state.attention[...,order]).abs().max())
        record("attention_row_sum",(state.attention.sum(-1)-1).abs().max(),2e-6)
        record("masked_attention",state.attention.masked_select((~obs.valid)[:,None,None,None,:]).abs().max(),0.)
        query_parts,retrieval_parts=[],[]
        for lo,hi in ((0,31),(31,79),(79,128)):
            part=QueryBatch(*(getattr(query,key)[:,lo:hi] for key in ("position","physical_time","time_present","quantity_id","valid")))
            features=a.query(part,specs,eq.context,encoded.summary)
            query_parts.append(features)
            retrieval_parts.append(a.retrieval(state,part,features,specs).features)
        record("query_chunks",(torch.cat(query_parts,1)-q_features).abs().max())
        record("retrieval_chunks",(torch.cat(retrieval_parts,1)-retrieved.features).abs().max())
        changed=replace(specs[1],bindings=tuple(replace(b,value=.04) if b.id=="alpha" else b for b in specs[1].bindings))
        original=a.equation(pack_graphs((specs[1],),vocabulary).to(resolved)).context
        modified=a.equation(pack_graphs((changed,),vocabulary).to(resolved)).context
        numeric_dependency=float((modified-original).abs().max())
        require(numeric_dependency>1e-6,"numeric coefficient has no embedding effect")
        altered_value=a.observation(replace(obs,value=obs.value+.3),specs)
        altered_position=a.observation(replace(obs,position=obs.position*.9),specs)
        value_dependency=float((altered_value.summary-encoded.summary).abs().max())
        position_dependency=float((altered_position.summary-encoded.summary).abs().max())
        require(min(value_dependency,position_dependency)>1e-4,"observation values/locations have no feature effect")
        dummy_time=obs.physical_time.clone();dummy_time[0]=float("nan")
        dummy=a.observation(replace(obs,physical_time=dummy_time),specs)
        record("stationary_dummy_time_inert",(dummy.tokens-encoded.tokens).abs().max(),0.)
        coordinate=CoordinateFeatures().to(resolved)
        valid=torch.ones(1,1,dtype=torch.bool,device=resolved)
        p=torch.full((1,1,2),.3,device=resolved);t=torch.zeros(1,1,1,device=resolved)
        absent=coordinate(p,t,~valid,valid);present=coordinate(p,t,valid,valid)
        require(not torch.equal(absent,present),"absent physical time collapsed into time=0")
        empty=replace(obs,valid=torch.zeros_like(obs.valid))
        empty_encoded=a.observation(empty,specs)
        empty_memory=a.memory(empty_encoded,eq.context,return_attention=True)
        require(bool(torch.isfinite(empty_memory.tokens).all()),"all-masked attention produced nonfinite output")
        record("all_masked_attention",empty_memory.attention.abs().max(),0.)
        graph_pad=a.equation(pack_graphs(specs,vocabulary,pad_nodes=4,pad_edges=5).to(resolved))
        record("graph_padding",(graph_pad.context-eq.context).abs().max())
    # Linear feature probes check derivative reachability only; no optimizer/update.
    a.zero_grad(set_to_none=True)
    numeric=graph.numeric.detach().requires_grad_()
    eq_grad=a.equation(replace(graph,numeric=numeric))
    tensors={name:getattr(obs,name).detach().requires_grad_() for name in ("value","sigma","position","physical_time")}
    obs_grad=a.observation(replace(obs,**tensors),specs)
    known=torch.ones(3,80,96,device=resolved,requires_grad=True)
    known_valid=torch.ones(3,80,dtype=torch.bool,device=resolved);known_valid[:,::2]=False
    mem_grad=a.memory(obs_grad,eq_grad.context,known_context=known,known_valid=known_valid)
    qp=query.position.detach().requires_grad_();qt=query.physical_time.detach().requires_grad_()
    q_grad=replace(query,position=qp,physical_time=qt)
    qf=a.query(q_grad,specs,eq_grad.context,obs_grad.summary)
    output=a.retrieval(mem_grad,q_grad,qf,specs)
    xi=torch.zeros(3,18,device=resolved,requires_grad=True)
    gradient=torch.ones_like(xi,requires_grad=True);previous=torch.zeros_like(xi,requires_grad=True)
    coeff_batch=coefficient_inputs(specs,xi,gradient,previous)
    coeff,coefficient_context=a.coefficient(coeff_batch)
    probe=output.features[...,0].mean()+mem_grad.tokens[...,0].mean()+eq_grad.context[:,1].mean()+coefficient_context[:,0].mean()
    backward_start=time.perf_counter()
    probe.backward()
    sync(resolved)
    backward_seconds=time.perf_counter()-backward_start
    derivative_inputs={**tensors,"graph_numeric":numeric,"known_context":known,"query_position":qp,"query_physical_time":qt,
                       "xi":xi,"current_gradient":gradient,"previous_update":previous}
    paths={name:{"finite":tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()),
                 "l1_norm":float(tensor.grad.abs().sum()) if tensor.grad is not None else 0.} for name,tensor in derivative_inputs.items()}
    require(all(p["finite"] and p["l1_norm"]>0 for p in paths.values()),"missing/nonfinite feature derivative path")
    record("stationary_time_gradient",tensors["physical_time"].grad[0].abs().max(),0.)
    record("stationary_query_time_gradient",qt.grad[0].abs().max(),0.)
    record("padded_value_gradient",tensors["value"].grad[~obs.valid].abs().max(),0.)
    record("inactive_coefficient_gradient",xi.grad[~coeff_batch.active].abs().max(),0.)
    parameter_gradients={name:{"finite":all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()),
                               "l1_norm":sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)}
                         for name,module in a.named_children()}
    require(all(p["finite"] and p["l1_norm"]>0 for p in parameter_gradients.values()),"unreachable representation parameters")
    attention_stats=[]
    weights=state.attention
    entropy=-(weights*weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
    for i,spec in enumerate(specs):
        attention_stats.append({"pde":spec.name,"valid_observations":int(obs.valid[i].sum()),
                                "mean_entropy_nats":float(entropy[i].mean()),"max_weight":float(weights[i].max())})
    # No hidden copies of an 80-token memory exist in the B or default C inventories.
    require(owners["B"].memory is None and owners["B"].retrieval is None and owners["C"].memory is None,"unexpected field memory allocation")
    sync(resolved)
    return {"passed":all(c["passed"] for c in checks),"device":str(resolved),"dtype":"float32",
            "elapsed_seconds":time.perf_counter()-began,"feature_probe_backward_seconds":backward_seconds,
            "normalizer_id":normalizers.identity,"vocabulary_id":vocabulary.identity,
            "shapes":{"graph_nodes":list(eq.nodes.shape),"equation_context":list(eq.context.shape),
                      "observation_tokens":list(encoded.tokens.shape),"observation_summary":list(encoded.summary.shape),
                      "memory":list(state.tokens.shape),"attention":list(state.attention.shape),
                      "query":list(q_features.shape),"retrieval":list(retrieved.features.shape),"coefficient_tokens":list(coeff.shape)},
            "parameters":inventories,"parameter_bytes_float32":{owner:4*sum(inv.values()) for owner,inv in inventories.items()},
            "estimates":computational_estimate(),"checks":checks,"derivative_paths":paths,"parameter_gradients":parameter_gradients,
            "numeric_coefficient_embedding_max_change":numeric_dependency,"observation_value_embedding_max_change":value_dependency,
            "observation_location_embedding_max_change":position_dependency,
            "attention_summary":attention_stats,"statistics":{"graph":stats(eq.nodes,graph.valid),"observations":stats(encoded.tokens,obs.valid),
                                                               "memory":stats(state.tokens),"retrieval":stats(retrieved.features)},
            "independent_parameter_storage":True,"global_RNG_preserved_on_construction":True,
            "limitations":["untrained feature modules; no scientific prediction or recovery results", "no A/B/C prediction architectures",
                           "V1 retrieval uses fixed-width all-anchor Gaussian weights, separately normalized per scale",
                           "B gradient/update inputs are diagnostic fixtures, not an implemented refinement algorithm"]}

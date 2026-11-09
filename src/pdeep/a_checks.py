"""Bounded structural A acceptance checks; no optimizer updates in this module."""
from dataclasses import fields,replace
import time
import torch
from .architecture_a import build_architecture_a
from .a_training import SupervisedBatch,objective,gradient_report
from .batches import QueryBatch,UnknownCoefficients
from .core import require,ValidationError
from .data import CaseDataset,observations,training_queries,collate_inputs,case_coefficients
from .reference import canonical_spec,evaluate_batch
from .representation_checks import padded,sync,stats
from .representations import float32_batch
from .encoding import pack_graphs


def slice_tokens(batch,lo,hi):
    return type(batch)(**{f.name:getattr(batch,f.name)[:,lo:hi] for f in fields(batch)})


def permute(batch,order):
    return type(batch)(**{f.name:getattr(batch,f.name)[:,order] for f in fields(batch)})


def mixed_fixture(normalizers,device="cpu",*,queries=100):
    dataset=CaseDataset(normalizers.root_seed)
    cases=tuple(dataset.case("train",name,7) for name in ("poisson","heat","wave"))
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    obs=collate_inputs([observations(dataset,c,normalizers,count=count,set_id="A-structural").inputs for c,count in zip(cases,(9,13,18))])
    query=collate_inputs([training_queries(dataset,c,count=queries,set_id="A-structural").inputs for c in cases])
    quantity=query.quantity_id.clone();quantity[2,1::2]=1
    query=replace(query,quantity_id=quantity)
    unknown=case_coefficients(cases);target=evaluate_batch(specs,unknown,query)
    return SupervisedBatch(specs,float32_batch(obs,device=device),float32_batch(query,device=device),target.to(device=device,dtype=torch.float32),
                           UnknownCoefficients(unknown.xi.to(device=device,dtype=torch.float32),unknown.active.to(device)),tuple(c.identity for c in cases))


def architecture_a_checks(normalizers,*,device="cpu"):
    start=time.perf_counter();a=build_architecture_a(normalizers).to(device).eval();b=mixed_fixture(normalizers,device)
    checks=[]
    def record(name,left,right,tol=2e-5):
        error=float((left-right).detach().abs().max()) if left.numel() else 0.
        checks.append({"name":name,"max_abs_error":error,"absolute_tolerance":tol,"passed":error<=tol})
    with torch.no_grad():
        output=a(b.specs,b.observations,b.queries,return_attention=True);context=output.context;memory=context.memory
        shapes={"graph_nodes":list(context.graph.nodes.shape),"graph_context":list(context.graph.context.shape),
                "observation_tokens":list(context.observations.tokens.shape),"memory":list(memory.tokens.shape),
                "attention":list(memory.attention.shape),"predictions":list(output.predicted_value.shape),"xi_hat":list(output.xi_hat.shape)}
        # B=2 with unequal N and common padding, independently from mixed gradient test.
        two=b.slice(0,2);out2=a(two.specs,two.observations,two.queries)
        require(out2.context.graph.context.shape==(2,96) and out2.context.observations.tokens.shape==(2,18,96)
                and out2.context.memory.tokens.shape==(2,80,96) and out2.predicted_value.shape==(2,100,1) and out2.xi_hat.shape==(2,18),"A1 tensor shapes")
        order=torch.arange(b.observations.valid.shape[1]-1,-1,-1,device=device)
        shuffled=a(b.specs,permute(b.observations,order),b.queries,return_attention=True)
        record("observation_permutation",output.predicted_value,shuffled.predicted_value)
        record("coefficient_permutation",output.xi_hat,shuffled.xi_hat)
        record("attention_permutation",memory.attention[...,order],shuffled.context.memory.attention)
        padded_obs=padded(b.observations,7)
        # Extreme finite fake values as well as NaN storage must both be inert.
        padded_obs=replace(padded_obs,value=torch.nan_to_num(padded_obs.value,nan=1e30),sigma=torch.nan_to_num(padded_obs.sigma,nan=-1e30))
        after=a(b.specs,padded_obs,b.queries,return_attention=True)
        record("observation_padding",output.predicted_value,after.predicted_value)
        record("coefficient_padding",output.xi_hat,after.xi_hat)
        record("attention_padding",memory.attention,after.context.memory.attention[...,:18])
        record("padded_attention_zero",after.context.memory.attention[...,18:],torch.zeros_like(after.context.memory.attention[...,18:]),0.)
        record("attention_valid_mass",memory.attention.sum(-1),torch.ones_like(memory.attention.sum(-1)),2e-6)
        memory_before=memory.tokens.clone()
        for name,sizes in (("chunks_10",list(range(0,101,10))),("chunks_uneven",[0,1,18,53,99,100])):
            pred=torch.cat([a.predict_queries(context,slice_tokens(b.queries,lo,hi)).predicted_value for lo,hi in zip(sizes[:-1],sizes[1:])],1)
            record(name,output.predicted_value,pred)
        individual=torch.cat([a.predict_queries(context,slice_tokens(b.queries,j,j+1)).predicted_value for j in range(100)],1)
        record("individual_queries",output.predicted_value,individual)
        record("context_immutable_across_chunks",memory.tokens,memory_before,0.)
        p=b.queries.position.clone();p[:,0]=.31
        changed=a.predict_queries(context,replace(b.queries,position=p)).predicted_value
        record("query_independence",output.predicted_value[:,1:],changed[:,1:],0.)
        dummy=b.queries.physical_time.clone();dummy[0]=float("nan")
        record("stationary_dummy_time",output.predicted_value,a.predict_queries(context,replace(b.queries,physical_time=dummy)).predicted_value,0.)
        boundary=b.queries.position.clone();boundary[:,0:4]=boundary.new_tensor([[0,.3],[1,.3],[.3,0],[.3,1]])
        pred_boundary=a.predict_queries(context,replace(b.queries,position=boundary)).predicted_value[:,:4]
        record("Dirichlet_boundary",pred_boundary,torch.zeros_like(pred_boundary),0.)
        query_pad=padded(b.queries,3)
        padpred=a.predict_queries(context,query_pad).predicted_value
        record("query_padding",padpred[:,:100],output.predicted_value)
        record("padded_predictions_zero",padpred[:,100:],torch.zeros_like(padpred[:,100:]),0.)
        empty_stats=[]
        for count in (0,1,8,18):
            obs=slice_tokens(b.observations,0,count)
            if count==0:obs=replace(obs,valid=torch.zeros_like(obs.valid))
            e=a(b.specs,obs,b.queries,return_attention=True)
            require(bool(torch.isfinite(e.predicted_value).all()),"nonfinite empty/multiple N output")
            empty_stats.append({"N":count,"finite":True,"max_abs":float(e.predicted_value.abs().max())})
        allmasked=replace(b.observations,valid=torch.zeros_like(b.observations.valid))
        e=a(b.specs,allmasked,b.queries,return_attention=True)
        record("empty_set_attention",e.context.memory.attention,torch.zeros_like(e.context.memory.attention),0.)
        record("empty_set_equivalence",e.predicted_value,a(b.specs,slice_tokens(allmasked,0,0),b.queries).predicted_value)
        sensitivities={}
        for name in ("value","position","physical_time","sigma","quantity_id"):
            value=getattr(b.observations,name).clone()
            if name=="value":value[2,0]+=.5
            elif name=="position":value[2,0]=.33
            elif name=="physical_time":value[2,0]=.47
            elif name=="sigma":value[2,0]*=2
            else:value[2,0]=1-value[2,0]
            altered=a(b.specs,replace(b.observations,**{name:value}),b.queries,return_attention=True)
            change=float((altered.predicted_value-output.predicted_value).abs().max())
            require(change>1e-8,f"A observation {name} pathway inert")
            sensitivities[name]={"prediction_max_abs_change":change,"attention_max_abs_change":float((altered.context.memory.attention-memory.attention).abs().max())}
        changed_heat=replace(b.specs[1],bindings=tuple(replace(v,value=.04) if v.id=="alpha" else v for v in b.specs[1].bindings))
        modified=a((b.specs[0],changed_heat,b.specs[2]),b.observations,b.queries)
        numeric_effect=float((output.predicted_value-modified.predicted_value).abs().max())
        require(numeric_effect>1e-8,"known scalar numeric pathway inert")
        # Functional memory dependency; this is a feature ablation, not a new model.
        ablations={}
        for label,level in (("coarse",0),("fine",1)):
            tokens=memory.tokens.clone();tokens[:,memory.levels==level]=0
            ablated=replace(context,memory=replace(memory,tokens=tokens))
            change=float((a.predict_queries(ablated,b.queries).predicted_value-output.predicted_value).abs().max())
            require(change>1e-8,f"unused {label} memory")
            ablations[label]=change
    # Actual objective gradient with all PDEs, u/v, no fixture feature-probe loss.
    a.train();a.zero_grad(set_to_none=True)
    out=a(b.specs,b.observations,b.queries);loss=objective(out,b);loss["total"].backward()
    gradients=gradient_report(a)
    require(all(g["l2_norm"]>0 and not g["missing"] and not g["nonfinite"] for g in gradients.values()),"unreachable A modules")
    raw=float(torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True))
    clipped=sum(float(p.grad.square().sum()) for p in a.parameters() if p.grad is not None)**.5
    require(clipped<=1.00001,"gradient clipping failed")
    # Field-only path stays connected; auxiliary head is correctly unused at weight 0.
    a.zero_grad(set_to_none=True);out=a(b.specs,b.observations,b.queries)
    objective(out,b,coefficient_weight=0)["field"].backward()
    field_gradients=gradient_report(a)
    require(all(g["l2_norm"]>0 for name,g in field_gradients.items() if name!="coefficient_head"),"field gradient path broken")
    # Continuous query derivatives near former nearest-neighbor switching lines.
    a.eval();context=a.prepare_context(b.specs,b.observations)
    pos=b.queries.position[:,:3].clone();pos[:,:,0]=pos.new_tensor([.5-1e-5,.5,.5+1e-5]);pos.requires_grad_()
    query=replace(slice_tokens(b.queries,0,3),position=pos)
    value=a.predict_queries(context,query).predicted_value
    derivative=torch.autograd.grad(value.sum(),pos)[0]
    h=2e-4
    plus=pos.detach().clone();minus=pos.detach().clone();plus[:,:,0]+=h;minus[:,:,0]-=h
    with torch.no_grad():fd=(a.predict_queries(context,replace(query,position=plus)).predicted_value-a.predict_queries(context,replace(query,position=minus)).predicted_value)/(2*h)
    scaled=(fd[...,0]-derivative[...,0]).abs()/(1+derivative[...,0].abs())
    record("coordinate_derivative_finite_difference_scaled",scaled,torch.zeros_like(scaled),.003)
    continuity=[]
    base=slice_tokens(b.queries,0,1)
    for eps in (1e-3,1e-4,1e-5,1e-6,1e-7):
        values=[];jacobians=[]
        for sign in (-1,1):
            point=base.position.detach().clone();point[...,0]=.5+sign*eps;point.requires_grad_()
            value=a.predict_queries(context,replace(base,position=point)).predicted_value
            jacobian=torch.autograd.grad(value.sum(),point)[0]
            values.append(value.detach());jacobians.append(jacobian.detach())
        continuity.append({"epsilon":eps,"value_gap":float((values[1]-values[0]).abs().max()),
                           "first_derivative_gap":float((jacobians[1]-jacobians[0]).abs().max())})
    # Conservative float32 bound at the finest spacing; no fixed jump remains.
    record("continuity_value_gap_at_1e-7",torch.tensor(continuity[-1]["value_gap"]),torch.tensor(0.),1e-5)
    record("continuity_derivative_gap_at_1e-7",torch.tensor(continuity[-1]["first_derivative_gap"]),torch.tensor(0.),1e-3)
    # Explicit rejected contracts.
    rejected=[]
    for label,bad in (("unknown_quantity",replace(b.queries,quantity_id=torch.full_like(b.queries.quantity_id,4))),
                      ("stationary_present_time",replace(b.queries,time_present=torch.ones_like(b.queries.time_present)))):
        try:a.predict_queries(context,bad)
        except ValidationError:rejected.append(label)
        else:raise AssertionError(label+" accepted")
    try:a.prepare_context(b.specs,replace(b.observations,sigma=torch.zeros_like(b.observations.sigma)))
    except ValidationError:rejected.append("noiseless")
    else:raise AssertionError("sigma=0 accepted")
    sync(next(a.parameters()).device)
    return {"passed":all(c["passed"] for c in checks),"device":device,"dtype":"float32","elapsed_seconds":time.perf_counter()-start,
            "shapes":shapes,"B2_shapes_passed":True,"parameters":a.parameter_inventory(),"total_parameters":sum(p.numel() for p in a.parameters()),
            "checks":checks,"sensitivities":sensitivities,"known_numeric_coefficient_prediction_change":numeric_effect,"memory_ablations":ablations,
            "gradient_norm_before_clip":raw,"gradient_norm_after_clip":clipped,"gradients":gradients,"field_only_gradients":field_gradients,
            "query_coordinate_derivative_max_abs":float(derivative.abs().max()),"continuity":continuity,"empty_observation_cases":empty_stats,"rejected":rejected,
            "statistics":{"graph":stats(output.context.graph.nodes),"observations":stats(output.context.observations.tokens,b.observations.valid),
                          "memory":stats(memory.tokens),"per_anchor_norms":memory.tokens.detach().norm(dim=-1).cpu().tolist(),
                          "padded_observation_fraction":float((~b.observations.valid).float().mean())},
            "interpretation":"untrained structural checks, no claim of physical sensitivity or learned accuracy"}

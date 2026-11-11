"""Separate arbitrary input sensitivity from genuine two-case conditioning skill."""
from dataclasses import replace
import torch
from ..core import digest
from .data import select,token_id
from .metrics import moments,scalar


def perturbations(data):
    obs=data.observations; n=obs.valid.shape[1]; out={}
    perm=torch.arange(n-1,-1,-1);out['complete_record_permutation']=select(obs,perm)
    def clone():return select(obs,torch.arange(n))
    v=clone();q=int(v.quantity_id[0,0]);v.value[0,0,0]+=.1*data.normalizers.scale(data.case.pde,q);out['one_value']=v
    v=clone();v.value[:,[0,1]]=v.value[:,[1,0]].clone();out['values_reassigned_to_positions']=v
    v=clone();v.position[0,0]=data.training.position[0,0];out['one_valid_location']=v
    v=clone();v.sigma[0,0]*=1.1;out['sigma_only']=v
    v=clone();v.valid[:,::2]=False;out['subset_removed']=v
    v=clone();v.valid[:]=False;out['empty_observations']=v
    if data.case.pde!='poisson':
        v=clone();v.physical_time[0,0,0]=.37 if obs.physical_time[0,0,0]<.5 else .63;out['physical_time']=v
    if data.case.pde=='wave':
        v=clone();v.quantity_id[0,0]=1-q;out['wave_quantity']=v
    return out


def sensitivity(model,data,paired_data=None,*,query_limit=128):
    device=next(model.parameters()).device; queries=select(data.diagnostic,torch.linspace(0,data.diagnostic.valid.shape[1]-1,min(query_limit,data.diagnostic.valid.shape[1])).long()).to(device)
    problem=(data.case.problem,); norms=data.normalizers
    def evaluate(obs):
        # Contexts ALWAYS rebuilt, even for repeated geometry/model weights.
        obs=obs.to(device);cs=model.prepare_context(problem,obs)
        tok,_,_=model.observation(cs[0].layout,problem[0],obs,0,cs[0].equation,norms)
        return model.predict(cs,queries),cs[0],tok
    previous=model.training;model.eval()
    try:
        with torch.no_grad():
            base,context,token=evaluate(data.observations); rows={}
            scales=base.new_tensor([norms.scale(data.case.pde,int(q)) for q in queries.quantity_id[0]])[None,:,None]
            for name,obs in perturbations(data).items():
                pred,c,tok=evaluate(obs);delta=pred-base
                # Pool tokens for unequal lengths; permutation test aligns complete records.
                compare=tok.flip(0) if name=='complete_record_permutation' else tok
                difference=scalar((compare-token).abs().mean()) if compare.shape==token.shape else scalar(((compare.mean(0) if len(compare) else torch.zeros_like(token.mean(0)))-token.mean(0)).abs().mean())
                rows[name]={'mean_absolute_difference':scalar(delta.abs().mean()),'max_absolute_difference':scalar(delta.abs().max()),'standardized_RMS_difference':scalar((delta/scales).square().mean().sqrt()),'conditioning_token_difference':difference,'global_context_difference':scalar((c.global_context-context.global_context).abs().mean()),'context_identity':digest((problem[0].identity,token_id(obs))),'query_identity':token_id(queries),'fresh_context':True}
        # Derivative of mean standardized prediction wrt a standardized sensor value.
        obs=select(data.observations,torch.arange(data.observations.valid.shape[1])).to(device);obs.value.requires_grad_(True)
        pred,_,_=evaluate(obs);g=torch.autograd.grad((pred/scales).mean(),obs.value)[0]
        q=int(obs.quantity_id[0,0]);s=norms.scale(data.case.pde,q);eps=.002*s
        with torch.no_grad():
            values=[]
            for sign in (-1,1):
                o=select(data.observations,torch.arange(obs.valid.shape[1]));o.value[0,0,0]+=sign*eps
                values.append(scalar((evaluate(o)[0]/scales).mean()))
        ad=scalar(g[0,0,0])*s;fd=(values[1]-values[0])/(2*.002)
        result={'perturbations':rows,'value_derivative':{'AD_standardized':ad,'FD_standardized':fd,'absolute_discrepancy':abs(ad-fd),'FD_standardized_step':.002,'all_sensor_gradient_RMS':scalar(g.square().mean().sqrt())},'interpretation':'Sensitivity alone does not establish physically correct conditioning. Permutation should be invariant; noise response has no guaranteed pointwise sign.'}
        if paired_data is not None:
            # Both genuine references evaluated at the SAME independent coordinates.
            from ..smooth.data import evaluate_pairs
            import numpy as np
            x=queries.position[0].detach().cpu().double().numpy();t=queries.physical_time[0,:,0].detach().cpu().double().numpy();q=queries.quantity_id[0].cpu().numpy()
            truth1=queries.value
            truth2=torch.as_tensor(evaluate_pairs(paired_data.case,x,t,q),device=device,dtype=truth1.dtype)[None,:,None]
            with torch.no_grad():
                p1,_,_=evaluate(data.observations);p2,_,_=evaluate(paired_data.observations)
                empty=perturbations(data)['empty_observations'];pe,_,_=evaluate(empty)
                shuffled=select(data.observations,torch.arange(data.observations.valid.shape[1]));shuffled.value=shuffled.value.flip(1);ps,_,_=evaluate(shuffled)
                predictions={'O1':p1,'O2':p2,'empty':pe,'Gaussian_prior_mean_zero':torch.zeros_like(p1),'shuffled_values_O1':ps}
                result['paired']={name:{'truth1_standardized_MSE':scalar(((pr-truth1)/scales).square().mean()),'truth2_standardized_MSE':scalar(((pr-truth2)/scales).square().mean())} for name,pr in predictions.items()}
                result['paired']['improvement_correct_vs_wrong']={'truth1':scalar((((p2-truth1)/scales).square()-((p1-truth1)/scales).square()).mean()),'truth2':scalar((((p1-truth2)/scales).square()-((p2-truth2)/scales).square()).mean())}
                result['paired_case_ids']=[data.case.case_id,paired_data.case.case_id]
        return result
    finally:model.train(previous)

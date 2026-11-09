"""Bounded B0/B1 physical, structural, gradient and ablation verification."""
from dataclasses import replace
import time
import torch
from .architecture_b import build_architecture_b,BAblation
from .b_physics import build_system,physical_solution,physical_batch,cpu64,fixed_refinement
from .b_training import coefficient_loss
from .a_checks import mixed_fixture,permute,slice_tokens
from .a_training import gradient_report
from .representation_checks import padded
from .bayes import ExactGaussianD
from .core import require


def inverse_math_checks(normalizers):
    b=mixed_fixture(normalizers,queries=12);system=build_system(b.specs,b.observations);checks=[]
    xi=system.mean+.31*system.scale*system.active;xi.requires_grad_()
    analytic=system.state(xi)[2];ad=torch.autograd.grad(system.state(xi)[-1].sum(),xi)[0]
    checks.append(dict(name='inverse_gradient_AD',max_abs_error=float((ad-analytic).detach().abs().max()),tolerance=1e-9))
    fd=torch.zeros_like(xi)
    for row in range(3):
        for j in torch.where(system.active[row])[0]:
            h=1e-5*float(system.scale[row,j]);plus=xi.detach().clone();minus=plus.clone();plus[row,j]+=h;minus[row,j]-=h
            fd[row,j]=(system.state(plus)[-1][row]-system.state(minus)[-1][row])/(2*h)
    error=((fd-analytic.detach()).abs()/(1+analytic.detach().abs()))[system.active].max()
    checks.append(dict(name='inverse_gradient_FD_scaled',max_abs_error=float(error),tolerance=1e-7))
    query=b.observations
    from .reference import evaluate_batch
    from .batches import UnknownCoefficients
    prediction=evaluate_batch(b.specs,UnknownCoefficients(xi,system.active),physical_batch(query))
    checks.append(dict(name='H_vs_validated_solution',max_abs_error=float((system.state(xi)[0]-prediction[...,0]).detach().abs().max()),tolerance=1e-12))
    for x in checks:x['passed']=x['max_abs_error']<=x['tolerance']
    return dict(passed=all(x['passed'] for x in checks),checks=checks,device='cpu',dtype='float64')


def estimate_summary(model,result,batch):
    system=result.context.system;truth=cpu64(batch.unknowns.masked());quantities=[]
    prediction=model.predict_queries(result,batch.queries)
    error=prediction.predicted_value.to('cpu').double()-cpu64(batch.target)
    for i,s in enumerate(batch.specs):
        values={}
        for q in (0,1):
            valid=(batch.queries.valid[i]&(batch.queries.quantity_id[i]==q)).to('cpu')
            if bool(valid.any()):values[str(q)]=float(error[i,valid].detach().square().mean().sqrt())
        quantities.append(dict(pde=s.name,field_RMSE_by_quantity=values,
                               initial_standardized_xi_RMSE=float((((result.trajectory[i,0]-truth[i])/system.scale[i])[system.active[i]]).detach().square().mean().sqrt()),
                               final_standardized_xi_RMSE=float((((result.xi_hat[i]-truth[i])/system.scale[i])[system.active[i]]).detach().square().mean().sqrt()),
                               objectives=result.objectives[i].detach().tolist(),sensor_noise_standardized_RMSE=(result.residuals[i]/system.sigma[i]).detach()[:,system.valid[i]].square().mean(1).sqrt().tolist()))
    return quantities


def gradient_details(model):
    report=gradient_report(model)
    if model.spatial is not None:
        report['spatial_modules']={name:dict(norm=sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)**.5,
                                           missing=[n for n,p in m.named_parameters() if p.grad is None]) for name,m in model.spatial.named_children()}
    return report


def architecture_b_checks(normalizers,*,device='cpu'):
    began=time.perf_counter();b=mixed_fixture(normalizers,device,queries=40);checks=[];variants={};models={}
    def record(name,a,z,tol=2e-5):
        e=float((a-z).detach().abs().max()) if a.numel() else 0.;checks.append(dict(name=name,max_abs_error=e,tolerance=tol,passed=e<=tol))
    for approach in ('B0','B1'):
        m=build_architecture_b(approach,normalizers).to(device).eval();models[approach]=m
        with torch.no_grad():
            context=m.prepare_context(b.specs,b.observations);r=m.infer_unknowns(context,return_diagnostics=True)
            repeated=m.infer_unknowns(context,return_diagnostics=True)
            record(approach+'_determinism',r.trajectory,repeated.trajectory,0.)
            shuffled=m.infer_unknowns(m.prepare_context(b.specs,permute(b.observations,torch.arange(17,-1,-1,device=device))))
            record(approach+'_permutation',r.xi_hat,shuffled.xi_hat)
            pad=m.infer_unknowns(m.prepare_context(b.specs,padded(b.observations,7)))
            record(approach+'_padding',r.xi_hat,pad.xi_hat)
            prediction=m.predict_queries(r,b.queries)
            record(approach+'_physical_solution',prediction.physical_value,physical_solution(context.system,r.xi_hat,b.queries),0.)
            parts=torch.cat([m.predict_queries(r,slice_tokens(b.queries,lo,hi)).physical_value for lo,hi in ((0,1),(1,11),(11,39),(39,40))],1)
            record(approach+'_query_chunks',parts,prediction.physical_value,1e-12)
            before=r.xi_hat.clone();changed=replace(b.queries,position=b.queries.position*.9)
            m.predict_queries(r,changed);record(approach+'_queries_do_not_change_xi',r.xi_hat,before,0.)
            record(approach+'_inactive_slots',r.trajectory.masked_select((~r.xi_active)[:,None,:]),torch.zeros_like(r.trajectory.masked_select((~r.xi_active)[:,None,:])),0.)
            increase=(r.objectives[:,1:]-r.objectives[:,:-1]).clamp_min(0).max()
            record(approach+'_bounded_step_descent',increase,torch.zeros_like(increase),1e-8)
            require(bool((r.diagonal[r.xi_active[:,None].expand_as(r.diagonal)]>0).all()) and bool((r.steps>0).all()),'nonpositive B update')
            empty=replace(b.observations,valid=torch.zeros_like(b.observations.valid))
            er=m.infer_unknowns(m.prepare_context(b.specs,empty))
            require(bool(torch.isfinite(er.xi_hat).all()),'empty B nonfinite')
        m.zero_grad(set_to_none=True);differentiable=m.infer_unknowns(m.prepare_context(b.specs,b.observations),return_diagnostics=True)
        coefficient_loss(differentiable,b.unknowns).backward();grad=gradient_details(m)
        require(all(g['l2_norm']>0 and not g['missing'] and not g['nonfinite'] for name,g in grad.items() if name!='spatial_modules'),'unreachable B modules')
        if approach=='B1':require(all(g['norm']>0 and not g['missing'] for g in grad['spatial_modules'].values()),'unreachable B spatial submodule')
        variants[approach]=dict(parameters=m.parameter_inventory(),total_parameters=sum(m.parameter_inventory().values()),gradients=grad,
                                cases=estimate_summary(m,r,b),trajectory_shape=list(r.trajectory.shape),residuals_shape=list(r.residuals.shape),
                                token_shape=list(r.token_history[0].shape),physical_construction_seconds=context.system.construction_seconds)
    require(not({p.data_ptr() for p in models['B0'].parameters()}&{p.data_ptr() for p in models['B1'].parameters()}),'shared B0/B1 parameters')
    # Functional ablations are forward/backward checks, NOT training runs.
    m=models['B1'];ablations={}
    for label,flags in (('no_spatial',BAblation(spatial=False)),('no_spatial_attention',BAblation(spatial_attention=False)),('no_physical_projections',BAblation(physical_projections=False))):
        m.zero_grad(set_to_none=True);r=m.infer_unknowns(m.prepare_context(b.specs,b.observations),ablation=flags,return_diagnostics=True)
        coefficient_loss(r,b.unknowns).backward()
        absent=[name for name,p in m.named_parameters() if name.startswith('spatial.') and p.grad is None]
        if not flags.spatial:require(all(p.grad is None for p in m.spatial.parameters()) and not r.spatial_history,'disabled spatial path contributes gradients/state')
        if not flags.spatial_attention:
            disabled=('residual_tokens','blocks.0.q','blocks.0.k','blocks.0.v','blocks.0.output','blocks.0.bias','blocks.0.norm.','blocks.1.q','blocks.1.k','blocks.1.v','blocks.1.output','blocks.1.bias','blocks.1.norm.')
            require(all(p.grad is None for name,p in m.spatial.named_parameters() if name.startswith(disabled)),'disabled attention gradients')
        if not flags.physical_projections:require(all(p.grad is None for p in m.spatial.physical.parameters()),'disabled physical projection gradients')
        ablations[label]=dict(cases=estimate_summary(m,r,b),inactive_parameter_names=absent)
    # Change an informative sensor; recurrent spatial state must actually respond.
    with torch.no_grad():
        base=m.infer_unknowns(m.prepare_context(b.specs,b.observations),return_diagnostics=True)
        value=b.observations.value.clone();value[2,0]+=.4
        pert=m.infer_unknowns(m.prepare_context(b.specs,replace(b.observations,value=value)),return_diagnostics=True)
        change=float((base.spatial_history[-1]['tokens']-pert.spatial_history[-1]['tokens']).abs().max())
        require(change>1e-5,'B1 spatial state insensitive to informative data')
    controls=[]
    for i,spec in enumerate(b.specs):
        # D CPU64 exact conditioning sees exactly the same rounded neural-input data.
        post=ExactGaussianD().fit(spec,physical_batch(b.observations),row=i)
        truth=cpu64(b.unknowns.masked())[i,:len(post.mean)]
        controls.append(dict(pde=spec.name,D_posterior_mean=post.mean.tolist(),
                             D_standardized_coefficient_RMSE=float(((post.mean-truth)/post.prior.covariance.diag().sqrt()).square().mean().sqrt())))
    system=build_system(b.specs,b.observations);fixed,fixed_obj=fixed_refinement(system)
    return dict(passed=all(x['passed'] for x in checks),device=device,neural_dtype='float32',physical_dtype='float64',physics_device='cpu',
                elapsed_seconds=time.perf_counter()-began,variants=variants,checks=checks,ablations=ablations,spatial_observation_change=change,
                fixed_control_objectives=fixed_obj.tolist(),D_controls=controls,matched_case_ids=list(b.case_ids),matched_H=True,
                interpretation='untrained bounded structural checks; B0/B1 accuracy differences are not scientific evidence')

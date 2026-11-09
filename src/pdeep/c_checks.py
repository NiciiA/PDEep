"""Bounded C structure, derivative, quadrature and objective connectivity checks."""
from dataclasses import replace
import time
import torch
from .architecture_c import build_architecture_c
from .a_checks import mixed_fixture,slice_tokens,permute
from .a_training import objective as supervised_objective,gradient_report
from .constraints import physical_losses,field_derivatives,strong_diagnostics
from .representation_checks import padded
from .core import require,ValidationError


def independent_diagnostic_queries(batch):
    """Keyed monitoring points, independent of Gauss nodes and supervised queries."""
    from .batches import QueryBatch
    from .diagnostics import RandomStreams
    rng=RandomStreams(20261108).torch('monitoring','C','strong-interior')
    b=len(batch.specs);p=.05+.9*torch.rand(b,24,2,generator=rng);t=torch.rand(b,24,1,generator=rng)
    valid=torch.ones(b,24,dtype=torch.bool);present=valid.clone()
    for i,s in enumerate(batch.specs):
        if s.name=='poisson':present[i]=False;t[i]=0
    q=QueryBatch(p,t,present,torch.zeros(b,24,dtype=torch.int64),valid)
    from .representations import float32_batch
    device=batch.queries.position.device
    return float32_batch(q,device=device.type if device.type in {'cpu','mps'} else str(device))


def serializable_strong(result):
    return [{k:([t.cpu().tolist() for t in v] if k=='residual_values' else v) for k,v in row.items()} for row in result]


def architecture_c_checks(normalizers,*,device='cpu'):
    start=time.perf_counter();c=build_architecture_c(normalizers).to(device).eval();b=mixed_fixture(normalizers,device,queries=40)
    checks=[]
    def record(name,a,z,tol=3e-5):
        error=float((a-z).detach().abs().max()) if a.numel() else 0.
        checks.append(dict(name=name,max_abs_error=error,tolerance=tol,passed=error<=tol))
    with torch.no_grad():
        output=c(b.specs,b.observations,b.queries,return_attention=True);ctx=output.context
        order=torch.arange(b.observations.valid.shape[1]-1,-1,-1,device=device)
        shuffled=c(b.specs,permute(b.observations,order),b.queries,return_attention=True)
        record('observation_permutation',output.predicted_value,shuffled.predicted_value)
        record('coefficient_permutation',output.xi_hat,shuffled.xi_hat)
        record('self_attention_permutation',ctx.attention[:,:, :,order][:,:,:,:,order],shuffled.context.attention)
        after=c(b.specs,padded(b.observations,7),b.queries,return_attention=True)
        record('observation_padding',output.predicted_value,after.predicted_value)
        record('conditioning_padding',ctx.conditioning,after.context.conditioning)
        record('masked_self_attention',ctx.attention.masked_select((~b.observations.valid)[:,None,None,None,:]),torch.zeros_like(ctx.attention.masked_select((~b.observations.valid)[:,None,None,None,:])),0.)
        masked_queries=ctx.attention.masked_select((~b.observations.valid)[:,None,None,:,None])
        record('masked_self_attention_queries',masked_queries,torch.zeros_like(masked_queries),0.)
        parts=[c.predict_queries(ctx,slice_tokens(b.queries,lo,hi)).predicted_value for lo,hi in ((0,1),(1,13),(13,39),(39,40))]
        record('query_chunks',torch.cat(parts,1),output.predicted_value)
        pos=b.queries.position.clone();pos[:,0]=.331
        record('query_independence',c.predict_queries(ctx,replace(b.queries,position=pos)).predicted_value[:,1:],output.predicted_value[:,1:],0.)
        times=b.queries.physical_time.clone();times[0]=float('nan')
        record('stationary_dummy_time',c.predict_queries(ctx,replace(b.queries,physical_time=times)).predicted_value,output.predicted_value,0.)
        boundary=b.queries.position.clone();boundary[:,:4]=boundary.new_tensor([[0,.2],[1,.2],[.3,0],[.3,1]])
        state=c.evaluate_state(ctx,replace(b.queries,position=boundary))
        record('boundary_u',state.u[:,:4],torch.zeros_like(state.u[:,:4]),0.)
        record('boundary_v',state.v[:,:4],torch.zeros_like(state.v[:,:4]),0.)
        empty=replace(b.observations,valid=torch.zeros_like(b.observations.valid))
        emptyout=c(b.specs,empty,b.queries,return_attention=True)
        emptyzero=c(b.specs,slice_tokens(empty,0,0),b.queries)
        record('empty_equivalence',emptyout.predicted_value,emptyzero.predicted_value)
        record('empty_attention',emptyout.context.attention,torch.zeros_like(emptyout.context.attention),0.)
        require(bool(torch.isfinite(emptyout.predicted_value).all()),'nonfinite C empty set')
        dependencies={}
        for name in ('value','position','physical_time','sigma','quantity_id'):
            value=getattr(b.observations,name).clone()
            if name=='value':value[2,0]+=.4
            elif name=='position':value[2,0]=.32
            elif name=='physical_time':value[2,0]=.413
            elif name=='sigma':value[2,0]*=2
            else:value[2,0]=1-value[2,0]
            changed=c(b.specs,replace(b.observations,**{name:value}),b.queries)
            effect=float((changed.predicted_value-output.predicted_value).abs().max());require(effect>1e-8,f'C observation {name} pathway inert');dependencies[name]=effect
        changed_heat=replace(b.specs[1],bindings=tuple(replace(v,value=.04) if v.id=='alpha' else v for v in b.specs[1].bindings))
        effect=c((b.specs[0],changed_heat,b.specs[2]),b.observations,b.queries)
        numeric_effect=float((effect.predicted_value-output.predicted_value).abs().max());require(numeric_effect>1e-8,'numeric C known pathway inert')
    # Value and derivative independence, and sum-gradient versus separate points.
    context=c.prepare_context(b.specs,b.observations)
    query=slice_tokens(b.queries,0,3)
    query=replace(query,position=query.position.clone().requires_grad_(),physical_time=query.physical_time.clone().requires_grad_())
    derivative=field_derivatives(c,context,query,create_graph=False,second=True)
    individually=[]
    for j in range(3):
        part=slice_tokens(query,j,j+1)
        individually.append(field_derivatives(c,context,part,create_graph=False,second=True))
    record('sum_gradient_equals_individual',derivative.grad_u,torch.cat([d.grad_u for d in individually],1))
    record('time_gradient_equals_individual',derivative.u_t,torch.cat([d.u_t for d in individually],1))
    changed=query.position.clone();changed[:,0]=.37
    # Compare the SAME derivative graph mode. MPS uses different rounding for
    # higher-order-capable LayerNorm backward versus its first-order kernel.
    d2=field_derivatives(c,context,replace(query,position=changed),create_graph=False,second=True)
    record('query_derivative_independence',derivative.grad_u[:,1:],d2.grad_u[:,1:],0.)
    untouched=c.evaluate_state(context,query).u[:,1].sum()
    cross=torch.autograd.grad(untouched,query.position)[0][:,[0,2]]
    record('off_diagonal_query_Jacobian',cross,torch.zeros_like(cross),0.)
    record('stationary_time_derivative',derivative.u_t[0],torch.zeros_like(derivative.u_t[0]),0.)
    h=2e-4;plus=query.position.detach().clone();minus=plus.clone();plus[...,0]+=h;minus[...,0]-=h
    with torch.no_grad():
        fd=(c.evaluate_state(context,replace(query,position=plus)).u-c.evaluate_state(context,replace(query,position=minus)).u)/(2*h)
    record('neural_spatial_FD_scaled',(fd[...,0]-derivative.grad_u[...,0]).abs()/(1+derivative.grad_u[...,0].abs()),torch.zeros_like(fd[...,0]),.003)
    # All five objectives use the real full quadrature and inferred xi, no truth substitution.
    paths={}
    for label in ('field','xi','weak','initial','combined'):
        c.zero_grad(set_to_none=True);out=c(b.specs,b.observations,b.queries)
        sup=supervised_objective(out,b)
        if label in ('weak','initial','combined'):physical=physical_losses(c,out.context,xi_hat=out.xi_hat)
        loss=sup[label] if label in ('field','xi') else physical.weak_per_case.mean() if label=='weak' else physical.initial_per_case.mean() if label=='initial' else sup['total']+.1*(physical.weak_per_case.mean()+physical.initial_per_case.mean())
        loss.backward();paths[label]=gradient_report(c)
        expected={'coefficient_head'} if label=='field' else {'field'} if label=='xi' else set()
        for name,g in paths[label].items():
            require(not g['nonfinite'],f'nonfinite C {label}/{name} gradient')
            if name not in expected:require(g['l2_norm']>0 and not g['missing'],f'missing C {label}/{name} gradient')
        paths[label]['expected_disconnected_modules']=sorted(expected)
    # Chunked integrals must agree BEFORE squaring; same all-eight-time objective.
    c.zero_grad(set_to_none=True);context=c.prepare_context(b.specs,b.observations)
    full=physical_losses(c,context,create_graph=False)
    chunks=physical_losses(c,context,spatial_chunk=73,create_graph=False)
    record('weak_spatial_chunks',full.weak_per_case,chunks.weak_per_case)
    record('initial_spatial_chunks',full.initial_per_case,chunks.initial_per_case)
    for i in range(3):
        for j,(a,z) in enumerate(zip(full.cases[i]['residuals'],chunks.cases[i]['residuals'])):record(f'assembled_integral_chunks_{i}_{j}',a,z)
    finer=physical_losses(c,context,spatial_order=32,create_graph=False)
    convergence=[]
    for i,s in enumerate(b.specs):
        convergence.append({'pde':s.name,'weak_loss_16':float(full.weak_per_case[i].detach()),'weak_loss_32':float(finer.weak_per_case[i].detach()),
                            'weak_loss_abs_difference':float((full.weak_per_case[i]-finer.weak_per_case[i]).detach().abs()),
                            'initial_loss_abs_difference':float((full.initial_per_case[i]-finer.initial_per_case[i]).detach().abs()),
                            'residual_integral_max_abs_difference':max(float((a-z).detach().abs().max()) for a,z in zip(full.cases[i]['residuals'],finer.cases[i]['residuals']))})
    strong=serializable_strong(strong_diagnostics(c,context,independent_diagnostic_queries(b)))
    physical_values=[{'pde':s.name,'weak_normalized':float(full.weak_per_case[i].detach()),'initial_normalized':float(full.initial_per_case[i].detach()),
                      'weak_physical_MSE':[float(x.detach()) for x in full.cases[i]['physical_weak_MSE']],
                      'initial_physical_MSE':[float(x.detach()) for x in full.cases[i]['physical_initial_MSE']]} for i,s in enumerate(b.specs)]
    for label,bad in [('invalid_quantity',replace(b.queries,quantity_id=torch.full_like(b.queries.quantity_id,9))),('Poisson_time_present',replace(b.queries,time_present=torch.ones_like(b.queries.time_present)))]:
        try:c.predict_queries(context,bad)
        except ValidationError:pass
        else:raise AssertionError(label+' accepted')
    return {'passed':all(x['passed'] for x in checks),'device':device,'dtype':'float32','elapsed_seconds':time.perf_counter()-start,
            'parameters':c.parameter_inventory(),'total_parameters':sum(c.parameter_inventory().values()),'checks':checks,'gradient_paths':paths,
            'shapes':{'graph_nodes':list(ctx.graph.nodes.shape),'graph_context':list(ctx.graph.context.shape),'tokens':list(ctx.interaction_tokens.shape),
                      'observation_context':list(ctx.observation_context.shape),'conditioning':list(ctx.conditioning.shape),'predictions':list(output.predicted_value.shape),'xi':list(output.xi_hat.shape)},
            'sensor_path_changes':dependencies,'known_numeric_coefficient_change':numeric_effect,'quadrature_convergence':convergence,
            'quadrature_timing':full.timing,'physical_values':physical_values,'strong_diagnostics':strong,
            'limitations':'Untrained neural-field diagnostics; finite nine-mode tests are not complete PDE enforcement; neural 16/32 errors measured, not guaranteed small.'}

"""Frozen B0 + Classical benchmark on an existing immutable validation plan.

All main solvers use the EXACT same B0 float32 observation copy promoted to CPU64.
Original-CPU64 D is retained as a separately labeled precision control. Timing is
warm coefficient/posterior inference, excluding data generation, loading, scoring
and common modal coordinate decoding. D includes its complete existing fit API.
"""
from pathlib import Path
import csv,json,time
import numpy as np
import torch
from .core import require
from .batches import masked_values
from .b_physics import cpu64,physical_batch,physical_solution
from .bayes import ExactGaussianD,observed_system,modal_prior
from .post_training_solvers import Quadratic,gradient_descent,conjugate_gradient,BUDGETS,work_counts
from .representations import float32_batch
from .reference import canonical_spec
from .b0_diagnosis import independent_matrix
from .evaluation_plan import EvaluationPlan
from .evaluation_inference import sync
from .notebook_workflows import load_selected
from .train_validation import model_identity
from .diagnostics import source_identity
from .artifact_io import sha256,read_manifest
from .artifact_metadata import scientific_identity
from .runtime import environment_report


def initializer(context,model):
    """Same existing initializer head/physical transform; audited against native B0."""
    z=model.initializer(torch.cat((context.graph.context,context.observations.summary),-1))
    s=context.system
    return masked_values(s.mean+s.scale*cpu64(z),s.active)


def system_quadratic(system):
    d=int(system.active[0].sum());v=system.valid[0]
    return Quadratic.build(system.H[0,v,:d].numpy(),system.y[0,v].numpy(),system.sigma[0,v].numpy(),
                           system.mean[0,:d].numpy(),system.scale[0,:d].numpy())


def observation_quadratic(spec,obs):
    H,y,sigma=observed_system(spec,obs);prior=modal_prior(spec)
    return Quadratic.build(H.numpy(),y.numpy(),sigma.numpy(),prior.mean.numpy(),prior.factor.diag().numpy())


def rounded_observations(obs):return physical_batch(float32_batch(obs,device='cpu'))


def _times(values):
    a=np.asarray(values,dtype=float)
    return {'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'min':float(a.min()),'max':float(a.max())}


def score(q,xi,truth,posterior,grams):
    error=xi-truth;diff=xi-posterior;g=q.independent_gradient(xi)
    field={name:{'physical_MSE':max(0.,float(error@G@error)),
                 'standardized_MSE':max(0.,float(error@G@error))/scale**2} for name,(G,scale) in grams.items()}
    return {'coefficient_standardized_MSE':float(np.mean((error/q.scale)**2)),
            'field_standardized_MSE':float(np.mean([r['standardized_MSE'] for r in field.values()])),
            'distance_to_D_L2':float(np.linalg.norm(diff)),
            'distance_to_D_standardized_RMSE':float(np.sqrt(np.mean((diff/q.scale)**2))),
            'Phi':q.phi(xi),'optimality_gap':max(0.,float(.5*diff@q.P@diff)),
            'gradient_norm':float(np.linalg.norm(g)),
            'standardized_gradient_norm':float(np.linalg.norm(g*q.scale)),
            'independent_relative_residual':float(np.linalg.norm(g)/max(np.linalg.norm(q.b),1.)),
            'field_quantities':field}


def aggregate(cases):
    output={}
    for pde in ('poisson','heat','wave'):
        group=[r for r in cases if r['pde']==pde];methods={}
        for name in group[0]['methods']:
            rows=[r['methods'][name] for r in group];keys=rows[0]['metrics']
            methods[name]={'cases':len(rows),'algorithm':rows[0]['algorithm'],
                'metrics':{k:{'mean':float(np.mean([r['metrics'][k] for r in rows])),
                              'p95':float(np.quantile([r['metrics'][k] for r in rows],.95)),
                              'max':float(np.max([r['metrics'][k] for r in rows]))} for k in keys if k!='field_quantities'},
                'iterations':_times([r['iterations'] for r in rows]),
                'counts':{k:_times([r['counts'][k] for r in rows]) for k in rows[0]['counts']},
                'timing':{k:{**_times([np.median(r['timing_samples'][k]) for r in rows]),
                              'all_repeat_p95':float(np.quantile([v for r in rows for v in r['timing_samples'][k]],.95))}
                          for k in rows[0]['timing_samples']},
                'field_quantities':{q:{k:float(np.mean([r['metrics']['field_quantities'][q][k] for r in rows]))
                                      for k in rows[0]['metrics']['field_quantities'][q]} for q in rows[0]['metrics']['field_quantities']}}
        output[pde]=methods
    return output


def run_benchmark(diagnosis_path,*,output,device='mps',repeats=7):
    require(isinstance(repeats,int) and repeats>=3,'at least three timing repeats required')
    root=Path(__file__).resolve().parents[2];start=time.perf_counter();diagnosis_path=Path(diagnosis_path).resolve()
    previous=json.loads(diagnosis_path.read_text());original=Path(previous['saved_evaluation'])
    saved=json.loads(original.read_text());plan=EvaluationPlan.load(original.parent/(previous['plan_identity']+'.json'))
    require(plan.split=='validation' and all(sum(e.pde==p for e in plan.cases)==50 for p in ('poisson','heat','wave')),'exact existing 50 validation cases/PDE required')
    checkpoint=Path(previous['checkpoint']);meta=read_manifest(checkpoint,architecture='B0')
    protected={str(p):sha256(p) for p in (checkpoint,*(checkpoint.parent/n for n in meta['files']))}
    model,norm,metadata,status,_=load_selected(checkpoint,'B0',device=device);require(status=='trained','selected trained B0 required')
    modes=[(m,m.training) for m in model.modules()];fingerprint=model_identity(model);model.eval()
    require(scientific_identity()==meta['scientific_source_files'],'scientific implementation differs from trained checkpoint')
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    lookup={c['physical_id']:c for c in previous['cases']};saved_lookup={c['case_id']:c for c in saved['scientific']['cases']['B0']};grams={};cases=[];decoder_times={}
    try:
        for pde in plan.pdes:
            grams[pde]={}
            for quantity in dict(plan.quantities)[pde]:
                query=plan.query(pde,quantity);M=independent_matrix(pde,query.position[0].numpy(),query.physical_time[0].numpy(),query.quantity_id[0].numpy())
                grams[pde]['v' if quantity else 'u']=(M.T@M/len(M),norm.scale(pde,quantity))
        specs={}
        for family,scaling in (('GD','identity'),('Jacobi GD','jacobi'),('prior-scaled GD','prior')):
            for budget in BUDGETS:specs[f'B0+Classical {family} {budget}']=(scaling,budget)
        for index,entry in enumerate(plan.cases):
            case,conditioning,_=plan.regenerate(entry,norm);obs=conditioning.inputs;spec=canonical_spec(entry.pde);d=len(case.tensor())
            earlier=lookup[entry.physical_id];require(earlier['conditioning_identity']==conditioning.metadata.identity,'observation identity changed')
            matching=rounded_observations(obs);q=observation_quadratic(spec,matching)
            post=ExactGaussianD().fit(spec,matching);D=post.mean.numpy();truth=case.tensor().numpy()
            model_device=next(model.parameters()).device
            def prepare():return model.prepare_context((spec,),float32_batch(obs,device=model_device.type if model_device.type in {'cpu','mps'} else str(model_device)))
            with torch.no_grad():
                context=prepare();init=initializer(context,model)[0,:d].numpy();native=model.infer_unknowns(context)
            saved_x=np.asarray(earlier['stages']['B0_k0']['xi']);native_x=native.xi_hat[0,:d].numpy()
            replay=float(np.max(np.abs(native.trajectory[0,:,:d].numpy()-np.array(saved_lookup[entry.physical_id]['refinement']['xi_by_iteration']))))
            require(np.max(np.abs(init-native.trajectory[0,0,:d].numpy()))<1e-14,'initializer utility differs from existing B0')
            require(np.max(np.abs(init-saved_x))<64*np.finfo(np.float32).eps*(1+np.max(np.abs(saved_x))),'initializer checkpoint replay discrepancy')
            # Validate objective and its gradient against unchanged production state and AD.
            x=torch.tensor(np.pad(init,(0,18-d))[None],dtype=torch.float64,requires_grad=True)
            _,_,g,phi=context.system.state(x);ad=torch.autograd.grad(phi.sum(),x)[0][0,:d].numpy()
            independent=q.independent_gradient(init);g_numpy=g[0,:d].detach().numpy()
            require(np.linalg.eigvalsh(q.P)[0]>0 and np.max(np.abs(q.P-q.P.T))<=128*np.finfo(float).eps*(1+np.max(np.abs(q.P))),'posterior precision must be SPD')
            audit={'precision_symmetry_scaled':float(np.max(np.abs(q.P-q.P.T))/(1+np.max(np.abs(q.P)))),
                   'gradient_vs_production_scaled':float(np.max(np.abs(independent-g_numpy))/(1+np.max(np.abs(independent)))),
                   'gradient_vs_AD_scaled':float(np.max(np.abs(independent-ad))/(1+np.max(np.abs(independent)))),
                   'Phi_relative':abs(q.phi(init)-float(phi.detach()))/(1+q.phi(init)),
                   'native_trajectory_replay_max_abs':replay,'initializer_replay_max_abs':float(np.max(np.abs(init-saved_x)))}
            require(max(audit['gradient_vs_production_scaled'],audit['gradient_vs_AD_scaled'],audit['Phi_relative'])<1e-11,'quadratic audit failed')
            answers={}
            for name,(scaling,budget) in specs.items():answers[name]=gradient_descent(q,init,budget,scaling=scaling)
            answers['B0+Classical CG']=conjugate_gradient(q,init,trace=True)
            answers['Classical CG from prior']=conjugate_gradient(q,q.mean,trace=True)
            for name in ('B0+Classical CG','Classical CG from prior'):
                solution=answers[name];residual=q.independent_gradient(solution['xi']);threshold=solution['residual_threshold']
                tolerance=128*np.finfo(float).eps*(1+np.linalg.norm(q.b)+np.linalg.norm(q.P,2)*np.linalg.norm(solution['xi']))
                require(solution['converged'] and np.linalg.norm(residual)<=threshold+tolerance,'independent CG residual verification failed')
                bound=(np.linalg.norm(residual)+tolerance)/np.linalg.eigvalsh(q.P)[0]
                require(np.linalg.norm(solution['xi']-D)<=bound,'CG displacement exceeds independent SPD residual bound')
                solution['independent_residual_norm']=float(np.linalg.norm(residual));solution['residual_error_bound']=float(bound)
            # Warm once per PDE/device; timing excludes model loading/compilation/data/scoring.
            if index==0 or plan.cases[index-1].pde!=entry.pde:
                with torch.no_grad():model.infer_unknowns(prepare());sync(model_device)
            if entry.pde not in decoder_times:
                coefficient=context.system.mean.clone();coefficient[0,:d]=torch.from_numpy(D)
                queries=[plan.query(entry.pde,k) for k in dict(plan.quantities)[entry.pde]]
                decoder_times[entry.pde]=[]
                for _ in range(repeats+1):
                    t=time.perf_counter()
                    with torch.no_grad():values=[physical_solution(context.system,coefficient,query) for query in queries]
                    decoder_times[entry.pde].append(time.perf_counter()-t)
                decoder_times[entry.pde]=decoder_times[entry.pde][1:]
            names=['B0 initializer','B0',*specs,'B0+Classical CG','Classical CG from prior','D matched B0 inputs','D original float64']
            timing={n:{'preparation':[],'initializer_head':[],'learned_refinement_including_head':[],
                       'classical_tail':[],'posterior_compute':[],'coefficient_total':[]} for n in names}
            def add_time(name,prep=0.,head=0.,learned=0.,tail=0.,posterior=0.,total=0.):
                for k,v in zip(timing[name],(prep,head,learned,tail,posterior,total)):timing[name][k].append(v)
            # Rotate method execution order deterministically across cases/repeats.
            for repeat in range(repeats):
                with torch.no_grad():
                    sync(model_device);t=time.perf_counter();ctx=prepare();sync(model_device);prep=time.perf_counter()-t
                    t=time.perf_counter();xi0=initializer(ctx,model)[0,:d].numpy();sync(model_device);head=time.perf_counter()-t
                    t=time.perf_counter();actual=model.infer_unknowns(ctx);sync(model_device);learned=time.perf_counter()-t
                add_time('B0 initializer',prep,head,total=prep+head)
                add_time('B0',prep,learned=learned,total=prep+learned)
                order=list(specs);shift=(index+repeat)%len(order);order=order[shift:]+order[:shift]
                for name in order:
                    scaling,budget=specs[name];t=time.perf_counter();qq=system_quadratic(ctx.system)
                    solved=gradient_descent(qq,xi0,budget,scaling=scaling);tail=time.perf_counter()-t
                    require(np.allclose(solved['xi'],answers[name]['xi'],rtol=1e-5,atol=1e-7),'timed solver/replay discrepancy')
                    add_time(name,prep,head,tail=tail,total=prep+head+tail)
                t=time.perf_counter();qq=system_quadratic(ctx.system);cg=conjugate_gradient(qq,xi0);tail=time.perf_counter()-t
                require(cg['converged'],'timed CG failed');add_time('B0+Classical CG',prep,head,tail=tail,total=prep+head+tail)
                for name in (['D matched B0 inputs','D original float64','Classical CG from prior'] if repeat%2 else ['Classical CG from prior','D original float64','D matched B0 inputs']):
                    t=time.perf_counter()
                    if name=='D original float64':fit=ExactGaussianD().fit(spec,obs)
                    elif name=='D matched B0 inputs':fit=ExactGaussianD().fit(spec,rounded_observations(obs))
                    else:
                        qq=observation_quadratic(spec,rounded_observations(obs));cg=conjugate_gradient(qq,qq.mean)
                    elapsed=time.perf_counter()-t
                    if name.startswith('D'):require(np.isfinite(fit.covariance.numpy()).all(),'nonfinite complete D posterior')
                    else:require(cg['converged'],'prior-start CG failed')
                    if name.startswith('D'):add_time(name,posterior=elapsed,total=elapsed)
                    else:add_time(name,tail=elapsed,total=elapsed)
            # This is a labeled additive estimate, NOT a separately measured full
            # pipeline: a shared analytic mean-field decoder for this fixed grid.
            for value in timing.values():
                value['estimated_with_shared_field_decode']=[a+b for a,b in zip(value['coefficient_total'],decoder_times[entry.pde])]
            rows={}
            def add(name,xi,algorithm,iterations,counts,extra=None):
                rows[name]={'algorithm':algorithm,'xi':xi.tolist(),'iterations':iterations,'counts':counts,
                            'metrics':score(q,xi,truth,D,grams[entry.pde]),'timing_samples':timing[name],**(extra or {})}
            zero=work_counts();zero.update(precision_builds=1,rhs_builds=0,eigensolves=1)
            add('B0 initializer',init,'frozen B0 initializer only',0,zero)
            learned_counts=dict(zero);learned_counts.update(objective_evaluations=4,gradient_evaluations=4,sensor_matvecs=8)
            add('B0',native_x,'unchanged B0, three learned refinements',3,learned_counts)
            for name,solution in answers.items():
                scaling=solution.get('scaling');kind='gradient descent, '+scaling+' scaling' if scaling else 'CG, posterior SPD precision'
                extra={'eta':solution.get('eta'),'positive_diagonal':solution['diagonal'].tolist() if scaling else None}
                if not scaling:
                    extra.update({k:solution[k] for k in ('converged','residual_norm','residual_threshold','independent_residual_norm','residual_error_bound')})
                    extra['iteration_metrics']=[{'iteration':k,**score(q,xi,truth,D,grams[entry.pde])} for k,xi in enumerate(solution['trajectory'])]
                counts=dict(solution['counts'])
                # Hybrids also pay the existing initializer's physical H/K preparation.
                if name.startswith('B0+'):
                    counts['precision_builds']+=1;counts['eigensolves']+=1
                add(name,solution['xi'],kind,solution['iterations'],counts,extra)
            full_d_counts=work_counts();full_d_counts.update(precision_builds=2,rhs_builds=1,eigensolves=2,prior_cholesky=1,
                posterior_cholesky=1,posterior_covariance_formations=1,covariance_triangular_solves=1,SVDs=2)
            add('D matched B0 inputs',D,'existing ExactGaussianD.fit: mean, covariance, factor, diagnostics',0,dict(full_d_counts),{'covariance_trace':float(np.trace(post.covariance.numpy()))})
            raw=ExactGaussianD().fit(spec,obs)
            add('D original float64',raw.mean.numpy(),'original-observation precision control; complete D fit',0,dict(full_d_counts),{'covariance_trace':float(np.trace(raw.covariance.numpy()))})
            # Independent coefficient->field risk agreement with the original saved result.
            cases.append({'pde':entry.pde,'case_id':entry.case_id,'physical_id':entry.physical_id,
                'case_identity':case.identity,'conditioning_identity':conditioning.metadata.identity,'audit':audit,
                'truth':truth.tolist(),'initial':init.tolist(),'posterior_mean':D.tolist(),
                'precision':{'dimension':d,'condition':float(np.linalg.cond(q.P)),
                             'jacobi_positive_diagonal':(1/np.diag(q.P)).tolist()},'methods':rows})
            if (index+1)%25==0:print(f'Frozen B0+Classical benchmark: {index+1}/150 cases ({repeats} warm repeats)',flush=True)
    finally:
        torch.set_num_threads(old_threads)
        for module,mode in modes:module.training=mode
    require(model_identity(model)==fingerprint and all(sha256(Path(p))==h for p,h in protected.items()),'model/checkpoint mutation detected')
    require(scientific_identity()==meta['scientific_source_files'],'scientific definitions changed')
    summary=aggregate(cases)
    paired={}
    rng=np.random.default_rng(20261110)
    for pde in plan.pdes:
        group=[c for c in cases if c['pde']==pde];paired[pde]={}
        for name in ('B0','B0+Classical CG','Classical CG from prior'):
            delta=np.array([np.median(c['methods'][name]['timing_samples']['coefficient_total'])-np.median(c['methods']['D matched B0 inputs']['timing_samples']['coefficient_total']) for c in group])
            bootstrap=rng.choice(delta,(2000,len(delta)),replace=True).mean(1)
            paired[pde][name]={'mean_time_difference_seconds':float(delta.mean()),'paired_case_bootstrap_CI95':np.quantile(bootstrap,[.025,.975]).tolist(),
                'median_time_ratio_to_D':summary[pde][name]['timing']['coefficient_total']['median']/summary[pde]['D matched B0 inputs']['timing']['coefficient_total']['median']}
    record={'version':'post-training-optimization-v1','plan_identity':plan.identity,'normalizer_identity':norm.identity,
        'checkpoint':str(checkpoint),'checkpoint_artifact_id':meta['artifact_id'],'checkpoint_files_sha256':protected,
        'original_result':str(original),'original_result_sha256':sha256(original),'previous_diagnosis':str(diagnosis_path),
        'device':str(next(model.parameters()).device),'classical_device':'CPU float64','runtime':environment_report(),
        'CPU_threads_during_benchmark':1,'thread_environment':{name:__import__('os').environ.get(name,'unset') for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},'repeats':repeats,'budgets':BUDGETS,'cases_per_PDE':{p:50 for p in plan.pdes},
        'summary':summary,'cases':cases,'paired_time_comparison_to_D':paired,'shared_full_grid_mean_decoder_seconds':decoder_times,'source':source_identity(root),'neural_training_updates':0,
        'weights_unchanged':True,'scientific_files_unchanged':len(scientific_identity()),'checkpoint_files_unchanged':True,
        'precision_policy':'All main methods share the EXACT B0 float32 observation copy promoted to CPU64. D original float64 is a separate original-benchmark control.',
        'timing_policy':'Warm coefficient/posterior inference. Repeats paired per case: sum measured preparation/head/tail components. Median/p95 across 50 case medians; all-repeat p95 also saved. Includes input transfer, H/prior setup, curvature/scaling setup and each solver. D includes complete existing fit (mean+covariance+factor+diagnostics). Excludes loading, dataset generation, mathematical verification and shared coordinate field decoding.',
        'GD_policy':'Physical coefficient GD, eta=1/lambda_max(P). Jacobi GD uses positive D=diag(P)^-1 (diagonal only), eta=1/lambda_max(sqrt(D) P sqrt(D)). Prior-scaled GD D=diag(s^2) is an additional no-learned-preconditioner control. Fixed budgets always execute every requested step.',
        'field_timing_note':'Shared analytic mean decoder measured per PDE on the fixed 64x64/all-times/all-quantities grid. estimated_with_shared_field_decode adds its measured samples to each coefficient solve; this is an explicitly labeled additive estimate, not separately timed end-to-end inference.',
        'CG_policy':'Unpreconditioned CG on physical SPD P x=b, starting frozen B0 initializer or prior mean control. rtol=atol=1e-12, maxiter=5*dimension; true residual recomputed before convergence. External sensor-form gradient/residual and SPD error bound verified.',
        'counts_policy':'Gradient/precision matvec counts are solver work. CG also counts true-residual recomputations. B0 uses 4 objectives+4 analytic gradients (8 sensor products). Matrix setup/eigensolves and D direct factorizations/covariance/SVD work are separate counters. External score verification is excluded from optimization counts/timing.',
        'elapsed_seconds':time.perf_counter()-start}
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True);(directory/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[]
    for c in cases:
        for name,r in c['methods'].items():
            flat.append({'pde':c['pde'],'case_id':c['case_id'],'method':name,'iterations':r['iterations'],
                **{k:v for k,v in r['metrics'].items() if k!='field_quantities'},**r['counts'],
                **{k+'_median_seconds':float(np.median(v)) for k,v in r['timing_samples'].items()}})
    with (directory/'per_case.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    return record

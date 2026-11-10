"""Read-only T1 B0 optimization audit; no model/physics/training changes.

Primary coefficient trajectories come from the saved matched evaluation, preserving
its exact observations and scores. A checkpoint replay audits learned D/eta. CPU64
SciPy trust-region least squares and NumPy augmented SVD solve the same quadratic,
independently of D's Cholesky posterior. Truth enters scoring only.
"""
from pathlib import Path
import csv,json,math,time
import numpy as np
import torch
from scipy.optimize import least_squares
from .core import require,digest
from .bayes import ExactGaussianD
from .b_physics import build_system,physical_batch
from .reference import canonical_spec
from .evaluation_plan import EvaluationPlan
from .evaluation_inference import Method
from .notebook_workflows import load_selected
from .train_validation import model_identity
from .diagnostics import source_identity
from .artifact_io import sha256,read_manifest
from .representations import float32_batch


def independent_matrix(pde,position,physical_time,quantity):
    """Independent NumPy scalar-mode formulas, including ordered wave a/b blocks."""
    xy=np.asarray(position,dtype=np.float64);t=np.asarray(physical_time).reshape(-1)
    q=np.asarray(quantity).reshape(-1);columns=[];velocity=[]
    for m in range(1,4):
        for n in range(1,4):
            lam=np.pi**2*(m*m+n*n);phi=2*np.sin(m*np.pi*xy[:,0])*np.sin(n*np.pi*xy[:,1])
            if pde=='poisson':columns.append(phi)
            elif pde=='heat':columns.append(phi*np.exp(-.02*lam*t))
            else:
                omega=.7*np.sqrt(lam);phase=omega*t
                columns.append(phi*np.where(q==1,-omega*np.sin(phase),np.cos(phase)))
                velocity.append(phi*np.where(q==1,np.cos(phase),np.sin(phase)/omega))
    return np.stack(columns+velocity,axis=1)


def independent_scales(pde):
    lam=np.array([np.pi**2*(m*m+n*n) for m in range(1,4) for n in range(1,4)])
    tau=.5*(lam[0]/lam)**.75
    return np.r_[tau,.4*.7*np.sqrt(lam)*tau] if pde=='wave' else tau


def classical_optimum(H,y,sigma,mean,scale,initial):
    """Minimize B Phi from xi0 in prior-standardized coordinates. No D input."""
    started=time.perf_counter()
    d=len(mean);A=H*scale[None]/sigma[:,None]
    J=np.vstack((A,np.eye(d)));target=np.r_[(y-H@mean)/sigma,np.zeros(d)]
    z0=(initial-mean)/scale
    result=least_squares(lambda z:J@z-target,z0,jac=lambda z:J,method='trf',tr_solver='exact',
                         loss='linear',x_scale=1.,ftol=None,xtol=1e-14,gtol=1e-11,max_nfev=100)
    # Independent direct least-squares cross-check, avoiding normal equations.
    z_svd=np.linalg.lstsq(J,target,rcond=None)[0]
    return mean+scale*result.x,{'success':bool(result.success),'status':int(result.status),'message':result.message,
            'nfev':result.nfev,'njev':result.njev,'seconds':time.perf_counter()-started,'cost':float(result.cost),'optimality':float(result.optimality),
            'vs_augmented_SVD_max_abs':float(np.max(np.abs(scale*(result.x-z_svd)))),
            'initial_xi':initial.tolist()},J,target


def _stats(values):
    a=np.asarray(values,dtype=np.float64)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),
            'p95':float(np.quantile(a,.95)),'max':float(a.max())}


def aggregate_stages(cases):
    output={}
    for pde in sorted({r['pde'] for r in cases}):
        rows=[r for r in cases if r['pde']==pde];stages={}
        for name in rows[0]['stages']:
            values=[r['stages'][name] for r in rows]
            keys=[k for k,v in values[0].items() if isinstance(v,(float,int))]
            stages[name]={k:_stats([v[k] for v in values]) for k in keys}
            # Root of mean case MSE, never the mean of case RMSEs.
            stages[name]['pooled_coefficient_RMSE']=float(np.sqrt(np.mean([v['coefficient_RMSE']**2 for v in values])))
            stages[name]['pooled_coefficient_standardized_RMSE']=float(np.sqrt(np.mean([v['coefficient_standardized_RMSE']**2 for v in values])))
            stages[name]['field']={q:{'physical_RMSE':float(np.sqrt(np.mean([v['field'][q]['MSE'] for v in values]))),
                        'standardized_MSE':float(np.mean([v['field'][q]['standardized_MSE'] for v in values]))}
                        for q in values[0]['field']}
            stages[name]['field_standardized_MSE']=float(np.mean([v['field_standardized_MSE'] for v in values]))
        output[pde]={'cases':len(rows),'stages':stages,
            'matrix':{k:_stats([r['matrix'][k] for r in rows]) for k in rows[0]['matrix'] if isinstance(rows[0]['matrix'][k],(float,int))},
            'audit':{k:max(r['audit'][k] for r in rows) for k in rows[0]['audit']},
            'refinement':{k:_stats([step[k] for r in rows for step in r['updates']]) for k in rows[0]['updates'][0] if isinstance(rows[0]['updates'][0][k],(float,int))},
            'Phi_monotone_cases':sum(all(b['Phi']<=a['Phi']+1e-9*(1+a['Phi']) for a,b in zip(list(r['stages'].values())[:3],list(r['stages'].values())[1:4])) for r in rows),
            'final_vs_initial_objective_gap_ratio':_stats([r['gap_ratio'] for r in rows]),
            'classical_optimizer_nfev':_stats([r['classical']['nfev'] for r in rows]),
            'classical_optimizer_successes':sum(r['classical']['success'] for r in rows)}
    return output


def investigate(result_path,*,output,device='mps',expected_cases_per_pde=50):
    """Audit a specific saved evaluation/export, never substitute another checkpoint."""
    started=time.perf_counter();result_path=Path(result_path).resolve();result=json.loads(result_path.read_text());science=result['scientific']
    plan=EvaluationPlan.load(result_path.parent/(science['plan_identity']+'.json'))
    require(plan.split=='validation' and set(plan.pdes)=={'poisson','heat','wave'},'audit requires the three T1 validation PDEs')
    counts={p:sum(e.pde==p for e in plan.cases) for p in plan.pdes}
    require(all(n==expected_cases_per_pde for n in counts.values()),'expected matched case count per PDE differs')
    require(len(science['cases']['B0'])==len(plan.cases),'missing B0 trajectories')
    checkpoint=Path(science['methods']['B0']['checkpoint']);manifest=read_manifest(checkpoint,architecture='B0')
    protected={str(p):sha256(p) for p in (checkpoint,*(checkpoint.parent/n for n in manifest['files']))}
    model,norm,meta,status,_=load_selected(checkpoint,'B0',device=device)
    require(status=='trained' or expected_cases_per_pde<50,'50-case diagnosis requires the selected trained B0 export')
    method=Method('B0','B0',status,model,str(checkpoint));fingerprint=model_identity(model)
    modes=[(m,m.training) for m in model.modules()];model.eval()
    lookup={r['case_id']:r for r in science['cases']['B0']};cache={};cases=[]
    try:
        for pde in plan.pdes:
            cache[pde]={}
            for quantity in dict(plan.quantities)[pde]:
                q=plan.query(pde,quantity);M=independent_matrix(pde,q.position[0].numpy(),q.physical_time[0].numpy(),q.quantity_id[0].numpy())
                cache[pde]['v' if quantity else 'u']=(M.T@M/len(M),norm.scale(pde,quantity))
        for index,entry in enumerate(plan.cases):
            case,cond,_=plan.regenerate(entry,norm);spec=canonical_spec(entry.pde);d=18 if entry.pde=='wave' else 9
            stored=lookup[entry.physical_id]
            require(stored['conditioning_identity']==entry.conditioning_identity and stored['case_identity']==case.identity,'saved result is not matched to plan')
            saved=np.array(stored['refinement']['xi_by_iteration']);truth=case.tensor().numpy()
            session=method.prepare(spec,cond.inputs);estimate=session.context;system=estimate.context.system
            H=system.H[0,system.valid[0],:d].numpy();y=system.y[0,system.valid[0]].numpy();sigma=system.sigma[0,system.valid[0]].numpy()
            mean=system.mean[0,:d].numpy();scale=system.scale[0,:d].numpy();valid=system.valid[0].numpy()
            rounded=physical_batch(float32_batch(cond.inputs,device='cpu'))
            H_ind=independent_matrix(entry.pde,rounded.position[0,valid].numpy(),rounded.physical_time[0,valid].numpy(),rounded.quantity_id[0,valid].numpy())
            prior=independent_scales(entry.pde);A=H*scale[None]/sigma[:,None];K=np.eye(d)+A.T@A;P=K/scale[:,None]/scale[None]
            b=H.T@(y/sigma**2)+mean/scale**2
            D_raw=ExactGaussianD().fit(spec,cond.inputs).mean.numpy();D_same=ExactGaussianD().fit(spec,rounded).mean.numpy()
            opt,optimizer,J,target=classical_optimum(H,y,sigma,mean,scale,saved[0])
            evals,eigenvectors=np.linalg.eigh(K);tol_z=128*np.finfo(float).eps*np.linalg.cond(K)*(1+np.max(np.abs((D_same-mean)/scale)))
            tol_x=float(np.max(scale)*tol_z)
            require(optimizer['success'] and np.max(np.abs(opt-D_same))<=tol_x,'classical optimizer disagrees with identical-input Gaussian optimum')
            require(optimizer['vs_augmented_SVD_max_abs']<=tol_x,'iterative optimizer disagrees with augmented SVD')
            audit={'H_independent_max_abs':float(np.max(np.abs(H-H_ind))),
                   'prior_scale_max_abs':float(np.max(np.abs(scale-prior))),
                   'prior_mean_max_abs':float(np.max(np.abs(mean))),
                   'y_roundtrip_max_abs':float(np.max(np.abs(y-rounded.value[0,valid,0].numpy()))),
                   'sigma_roundtrip_max_abs':float(np.max(np.abs(sigma-rounded.sigma[0,valid,0].numpy()))),
                   'curvature_relative_error':abs(float(system.curvature[0])-evals[-1])/evals[-1],
                   'checkpoint_replay_trajectory_max_abs':float(np.max(np.abs(estimate.trajectory[0,:,:d].numpy()-saved))),
                   'classical_vs_D_identical_inputs_max_abs':float(np.max(np.abs(opt-D_same))),
                   'classical_vs_D_raw_max_abs':float(np.max(np.abs(opt-D_raw))),
                   'classical_vs_SVD_max_abs':optimizer['vs_augmented_SVD_max_abs'],
                   'gradient_AD_scaled_error':0.,'gradient_FD_scaled_error':0.,'gradient_numpy_scaled_error':0.,
                   'objective_relative_error':0.,'recurrence_max_abs':0.,'saved_field_standardized_MSE_discrepancy':0.}
            raw=build_system((spec,),cond.inputs);Hr=raw.H[0,raw.valid[0],:d].numpy();yr=raw.y[0,raw.valid[0]].numpy();sr=raw.sigma[0,raw.valid[0]].numpy()
            Pr=Hr.T@(Hr/sr[:,None]**2)+np.diag(1/scale**2);br=Hr.T@(yr/sr**2)+mean/scale**2
            opt_raw,optimizer_raw,_,_=classical_optimum(Hr,yr,sr,mean,scale,saved[0])
            require(optimizer_raw['success'] and np.max(np.abs(opt_raw-D_raw))<=tol_x,'raw-input optimizer disagrees with D')
            audit['classical_raw_vs_D_raw_max_abs']=float(np.max(np.abs(opt_raw-D_raw)))
            fixed=saved[0].copy()
            for _ in range(3):fixed-=.95*scale**2*(P@fixed-b)/evals[-1]
            audit['B0_vs_fixed_095_three_steps_max_abs']=float(np.max(np.abs(fixed-saved[3])))
            stages={}
            for name,xi in list(zip(('B0_k0','B0_k1','B0_k2','B0_k3'),saved))+[('classical',opt),('D',D_raw),('D_identical_inputs',D_same),('classical_raw_inputs',opt_raw)]:
                padded=np.pad(xi,(0,18-d));x=torch.tensor(padded[None],dtype=torch.float64,requires_grad=True)
                _,_,grad,obj=system.state(x);g=grad[0,:d].detach().numpy();ad=torch.autograd.grad(obj.sum(),x)[0][0,:d].numpy()
                residual=(H@xi-y)/sigma;z=(xi-mean)/scale;phi=.5*(residual@residual+z@z)
                gn=H.T@((H@xi-y)/sigma**2)+(xi-mean)/scale**2
                audit['gradient_AD_scaled_error']=max(audit['gradient_AD_scaled_error'],float(np.max(np.abs(g-ad)/(1+np.abs(g)))))
                audit['gradient_numpy_scaled_error']=max(audit['gradient_numpy_scaled_error'],float(np.max(np.abs(g-gn))/(1+np.max(np.abs(g)))))
                audit['objective_relative_error']=max(audit['objective_relative_error'],abs(float(obj.detach())-phi)/(1+phi))
                # Central differences on the actual production Phi, standardized-coordinate h.
                if name.startswith('B0_'):
                    h=1e-5;fd=[]
                    for j in range(d):
                        plus=x.detach().clone();minus=plus.clone();plus[0,j]+=h*scale[j];minus[0,j]-=h*scale[j]
                        fd.append(float((system.state(plus)[-1]-system.state(minus)[-1])/(2*h)))
                    audit['gradient_FD_scaled_error']=max(audit['gradient_FD_scaled_error'],float(np.max(np.abs(np.array(fd)-g*scale))/(1+np.max(np.abs(g*scale)))))
                error=xi-truth;diff=xi-D_raw;fields={}
                spectral_error=eigenvectors.T@((xi-D_same)/scale);energy=float(spectral_error@spectral_error)
                raw_residual=(Hr@xi-yr)/sr;raw_gradient=Hr.T@((Hr@xi-yr)/sr**2)+(xi-mean)/scale**2

                for q,(G,s) in cache[entry.pde].items():
                    mse=max(0.,float(error@G@error));fields[q]={'MSE':mse,'RMSE':math.sqrt(mse),'standardized_MSE':mse/s**2}
                stages[name]={'xi':xi.tolist(),'coefficient_error':error.tolist(),'coefficient_RMSE':float(np.sqrt(np.mean(error**2))),
                    'coefficient_standardized_RMSE':float(np.sqrt(np.mean((error/scale)**2))),
                    'coefficient_a_RMSE':float(np.sqrt(np.mean(error[:9]**2))),
                    **({'coefficient_b_RMSE':float(np.sqrt(np.mean(error[9:]**2)))} if d==18 else {}),
                    'distance_to_D_L2':float(np.linalg.norm(diff)),'distance_to_D_standardized_RMSE':float(np.sqrt(np.mean((diff/scale)**2))),
                    'Phi':phi,'Phi_raw_objective':float(.5*(raw_residual@raw_residual+z@z)),
                    'gradient_norm_raw_objective':float(np.linalg.norm(raw_gradient)),
                    'slowest_half_modes_posterior_error_fraction':float(spectral_error[:d//2]@spectral_error[:d//2]/energy) if energy>1e-20 else 0.,
                    'Phi_gap':float(.5*((xi-D_same)@P@(xi-D_same))),
                    'gradient_norm':float(np.linalg.norm(g)),'standardized_gradient_norm':float(np.linalg.norm(g*scale)),
                    'field':fields,'field_standardized_MSE':float(np.mean([v['standardized_MSE'] for v in fields.values()]))}
            audit['saved_field_standardized_MSE_discrepancy']=abs(stages['B0_k3']['field_standardized_MSE']-np.mean([v['standardized_MSE'] for v in stored['quantities'].values()]))
            updates=[]
            for k in range(3):
                xi=estimate.trajectory[0,k,:d].numpy();g=estimate.gradients[0,k,:d].numpy()
                diagonal=estimate.diagonal[0,k,:d].numpy();eta=float(estimate.steps[0,k]);learned=diagonal/scale**2
                pre=np.sqrt(diagonal)[:,None]*P*np.sqrt(diagonal)[None];e=np.linalg.eigvalsh(pre)
                direction=diagonal*g;eta_opt=float(g@direction/(direction@P@direction)) if np.linalg.norm(direction)>0 else 0.
                audit['recurrence_max_abs']=max(audit['recurrence_max_abs'],float(np.max(np.abs(estimate.trajectory[0,k+1,:d].numpy()-xi+eta*diagonal*g))))
                updates.append({'k':k,'eta':eta,'fraction':eta*1.75*float(system.curvature[0]),'D_min':float(diagonal.min()),'D_max':float(diagonal.max()),
                    'learned_d_min':float(learned.min()),'learned_d_max':float(learned.max()),'learned_d_std':float(learned.std()),
                    'effective_step_eigen_min':float(eta*e[0]),'effective_step_eigen_max':float(eta*e[-1]),
                    'preconditioned_condition':float(e[-1]/e[0]),'slowest_contraction':float(1-eta*e[0]),
                    'exact_line_search_eta':eta_opt,'eta_over_line_search_eta':eta/eta_opt if eta_opt else 0.,
                    'diagonal':diagonal.tolist(),'learned_d':learned.tolist()})
            require(audit['H_independent_max_abs']<1e-12 and audit['prior_scale_max_abs']<1e-14,'H/prior mismatch')
            require(audit['gradient_AD_scaled_error']<1e-11 and audit['gradient_FD_scaled_error']<1e-7 and audit['gradient_numpy_scaled_error']<1e-11,'gradient audit failed')
            require(audit['checkpoint_replay_trajectory_max_abs']<64*np.finfo(np.float32).eps*(1+np.max(np.abs(saved))),'checkpoint replay differs beyond float32 tolerance')
            require(audit['recurrence_max_abs']<1e-12 and audit['saved_field_standardized_MSE_discrepancy']<1e-11,'recurrence or matched field metric audit failed')
            require(all(.25-2e-7<=u['learned_d_min']<=u['learned_d_max']<=1.75+2e-7 and .05-2e-7<=u['fraction']<=.95+2e-7 and 0<u['effective_step_eigen_max']<1 for u in updates),'step/preconditioner bounds violated')
            # Precision-induced optimum displacement, separately from optimizer error.
            forcing=(b-br)-(P-Pr)@D_raw
            precision_bound=float(np.linalg.norm(forcing)/np.linalg.eigvalsh(P)[0])
            require(np.linalg.norm(opt-D_raw)<=precision_bound+math.sqrt(d)*tol_x,'raw-vs-rounded posterior discrepancy exceeds perturbation bound')
            # Diagnostic only: three nonlearned steepest-descent steps with exact
            # quadratic line search, initialized from the same saved xi0.
            ls=saved[0].copy()
            for _ in range(3):
                g=P@ls-b;direction=scale**2*g;den=direction@P@direction
                if den:ls-=float(g@direction/den)*direction
            gap_ls=float(.5*(ls-D_same)@P@(ls-D_same))
            case_record={'pde':entry.pde,'case_id':entry.case_id,'physical_id':entry.physical_id,
                'case_identity':case.identity,'conditioning_identity':cond.metadata.identity,'valid_sensors':len(y),
                'truth':truth.tolist(),'D_mean':D_raw.tolist(),'D_identical_inputs_mean':D_same.tolist(),
                'stages':stages,'updates':updates,'classical':optimizer,'classical_raw_inputs':optimizer_raw,'audit':audit,
                'optimizer_coefficient_tolerance':tol_x,'float32_observation_posterior_L2_bound':precision_bound,
                'gap_ratio':stages['B0_k3']['Phi_gap']/stages['B0_k0']['Phi_gap'],
                'three_exact_line_search_steps_gap_ratio':gap_ls/stages['B0_k0']['Phi_gap'],
                'matrix':{'rank':int(np.linalg.matrix_rank(H)),'dimension':d,'sigma_min':float(sigma.min()),'sigma_max':float(sigma.max()),
                    'H_condition':float(np.linalg.cond(H)),'physical_precision_condition':float(np.linalg.cond(P)),
                    'prior_whitened_precision_condition':float(evals[-1]/evals[0]),'curvature':float(evals[-1]),
                    'whitened_precision_eigen_min':float(evals[0]),'whitened_precision_eigen_max':float(evals[-1]),
                    'whitened_precision_eigenvalues':evals.tolist(),'H_singular_values':np.linalg.svd(H,compute_uv=False).tolist()}}
            cases.append(case_record)
            if (index+1)%25==0:print(f'B0 read-only audit: {index+1}/{len(plan.cases)} matched cases',flush=True)
    finally:
        for module,mode in modes:module.training=mode
    require(model_identity(model)==fingerprint and all(sha256(Path(p))==h for p,h in protected.items()),'checkpoint/model mutation detected')
    from .artifact_metadata import scientific_identity
    require(scientific_identity()==meta['scientific_source_files'],'scientific implementation changed from the trained checkpoint')
    summary=aggregate_stages(cases)
    record={'version':'B0-T1-readonly-diagnosis-v1','source':source_identity(Path(__file__).resolve().parents[2]),
        'saved_evaluation':str(result_path),'saved_evaluation_sha256':sha256(result_path),'plan_identity':plan.identity,
        'checkpoint':str(checkpoint),'checkpoint_artifact_id':manifest['artifact_id'],'checkpoint_files_sha256':protected,
        'checkpoint_successful_updates':meta['successful_updates'],'neural_device':str(method.device),'reference_device':'CPU float64',
        'cases_per_PDE':counts,'summary':summary,'cases':cases,
        'macro_standardized_field_MSE':{stage:float(np.mean([r['stages'][stage]['field_standardized_MSE'] for r in summary.values()])) for stage in cases[0]['stages']},
        'original_macro_standardized_field_MSE':{k:v['macro']['standardized_MSE'] for k,v in science['aggregate'].items()},
        'metric_reduction':'points -> quantities equally per case -> cases equally per PDE -> PDEs equally; pooled RMSE=sqrt(mean MSE)',
        'precision_policy':'B0 H/y/sigma use explicit float32 observation copy then CPU64; D uses original CPU64. D_identical_inputs isolates this rounding.',
        'objective':'0.5 sum_valid((Hxi-y)/sigma)^2 + 0.5 sum_active((xi-mean)/scale)^2; no likelihood constants',
        'optimizer':'SciPy least_squares TRF, dense exact trust-region subproblem, augmented residual/Jacobian, starting saved B0 xi0; independent NumPy augmented SVD check',
        'optimizer_tolerance':'128*eps64*cond(K)*(1+norm(z_D,inf))*max(prior_scale), case-specific; raw D additionally bounded by coefficient perturbation norm/lambda_min(P)',
        'weights_unchanged':True,'checkpoint_files_unchanged':True,'scientific_compatibility_files_unchanged':len(meta['scientific_source_files']),'optimizer_training_updates':0,'architectures_changed':False,
        'elapsed_seconds':time.perf_counter()-started}
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    (directory/'diagnosis.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[]
    for r in cases:
        for stage,v in r['stages'].items():
            flat.append({'pde':r['pde'],'case_id':r['case_id'],'stage':stage,**{k:x for k,x in v.items() if isinstance(x,(float,int))},
                **{f'field_{q}_RMSE':f['RMSE'] for q,f in v['field'].items()}})
    with (directory/'per_case_stages.csv').open('w') as stream:
        keys=list(dict.fromkeys(k for row in flat for k in row));writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    return record

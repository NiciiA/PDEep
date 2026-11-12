"""Three focused studies: configure the existing suite/runner, no new engine.

All defaults are previews. Stable suite IDs and existing verified snapshots avoid
restarting completed arms. Explicit resume manifests work across Colab sessions.
"""
from dataclasses import asdict
from pathlib import Path
import json
from ..core import digest,require
from ..artifact_io import read_manifest,atomic_json
from .suite import SuiteSettings,prepare_diagnostic_suite,run_diagnostic_suite
from .engine import diagnostic_source
from .study_controls import C_VARIANTS,HEAT_VARIANTS,STAGED


def prepare_study(study,*,certificate,root,persistence=None,capacity='medium',device='auto',query_mode='dense_resampled',query_count=16384,
                  query_chunk=4096,query_time_policy='stratified',updates=1000,learning_rate=3e-4,seed=42,amp_mode='auto',memory_target_gb=11.,
                  action='fresh',resume_manifest=None,variants=None,staged=False,best_clip=None,advanced=None):
    require(study in {'c_wave','a_heat'},'select c_wave/a_heat')
    controls={k:dict(v,gradient_diagnostics_every=200) for k,v in (C_VARIANTS if study=='c_wave' else HEAT_VARIANTS).items()}
    if variants is not None:
        require(variants and set(variants)<=set(controls),'unknown/empty active study variants')
        controls={k:controls[k] for k in variants}
    if staged:
        require(study=='c_wave','staged physics is a C-wave diagnostic')
        controls['STAGED_001']={**STAGED,'gradient_diagnostics_every':200}
    if best_clip is not None:
        require(study=='a_heat' and best_clip in {1.,2.,5.},'explicit TRAIN-selected best_clip must be1/2/5')
        controls['EARLY_TIME_BEST_CLIP']={'grad_clip_norm':best_clip,'query_time_policy':'early_stratified_v1','gradient_diagnostics_every':200}
    adv=dict(advanced or {})
    if study=='c_wave':adv.update(pde='wave',family='square')
    settings=SuiteSettings(query_mode=query_mode,query_count=query_count,query_time_policy=query_time_policy,query_sampler_seed=seed,
        updates=updates,learning_rate=learning_rate,device=device,amp_mode=amp_mode,seed=seed,query_chunk=query_chunk,
        query_checkpointing=False,memory_target_gb=memory_target_gb,advanced=adv,variant_settings=controls)
    architecture='C_GEO' if study=='c_wave' else 'A_GEO';experiment='c_physics_ablation' if study=='c_wave' else 'heat_single'
    identifier='optimization-'+study+'-'+digest((asdict(settings),architecture,capacity,certificate['certification_id'],diagnostic_source()))[:20]
    # This is a pointer to existing suite artifacts, not a second persistence
    # format/transaction. Prefer a caller-selected verified Drive snapshot.
    if resume_manifest is None:
        local=Path(root)/'learning_diagnostics'/'suites'/identifier/'latest.json'
        if local.exists():
            info=json.loads(local.read_text());resume_manifest=info.get('manifest')
            if not resume_manifest or not Path(resume_manifest).is_file():resume_manifest=info.get('Drive_manifest')
        elif persistence is not None and persistence.drive_enabled:
            base=Path(persistence.drive_root)/'checkpoints' if persistence.drive_root else Path(persistence.drive_mount)/'PDEep_artifacts'
            available=list((base/identifier).glob('*/suite-state-*.json'))
            if available:resume_manifest=max(available,key=lambda p:int(p.stem.rsplit('-',1)[1]))
    effective='resume' if resume_manifest is not None else action
    plan=prepare_diagnostic_suite((experiment,),model=architecture,capacity=capacity,settings=settings,certificate=certificate,root=root,
        persistence=persistence,action=effective,resume_manifest=resume_manifest,suite_name=identifier)
    if effective=='fresh':
        # Reuse only completed arms from a genuinely matched earlier study.
        # This makes adding STAGED/combined arms avoid repeating fixed controls.
        from .suite import _snapshot
        candidates=list((Path(root)/'learning_diagnostics'/'suites').glob('optimization-'+study+'-*/latest.json'))
        manifests=[json.loads(p.read_text())['manifest'] for p in candidates]
        if persistence is not None and persistence.drive_enabled:
            base=Path(persistence.drive_root)/'checkpoints' if persistence.drive_root else Path(persistence.drive_mount)/'PDEep_artifacts'
            for folder in base.glob('optimization-'+study+'-*'):
                snapshots=list(folder.glob('*/suite-state-*.json'))
                if snapshots:manifests.append(max(snapshots,key=lambda p:int(p.stem.rsplit('-',1)[1])))
        reused={};wanted={k:v for k,v in asdict(settings).items() if k!='variant_settings'}
        for manifest in manifests:
            state,_=_snapshot(manifest);preview=state['preview'];old_settings=preview['settings']
            if state.get('source')!=plan.source or state['certificate']!=certificate['certification_id']:continue
            if settings.amp_mode=='auto' and (len(controls)==1)!=(len(old_settings.get('variant_settings',{}))==1):continue
            if digest({k:v for k,v in old_settings.items() if k!='variant_settings'})!=digest(wanted):continue
            for entry in state['entries'].values():
                for label,run in entry.get('runs',{}).items():
                    if run.get('status')=='completed' and label in controls and digest(old_settings.get('variant_settings',{}).get(label))==digest(controls[label]):
                        # Case, source, full checkpoint/runtime verification is
                        # repeated by the normal suite runner before any update.
                        from .suite import _checkpoint_path
                        meta=read_manifest(_checkpoint_path(run),kind='training_checkpoint',architecture=architecture)
                        if meta['config']['capacity']!=capacity or run['case_id']!=plan.recipes[0]['physical_case']:continue
                        reused[label]={**run,'reused_from':str(manifest)}
        plan.reused_runs=reused
    return plan



def run_study(plan,*,run_mode='preview',notebook=True):
    if run_mode=='preview':
        if notebook:
            print(f"{plan.model}: {len(plan.settings.variant_settings)} matched variants · {plan.settings.query_count:,} total queries/update · {plan.settings.updates} updates each · PREVIEW, zero optimizer updates.")
            print(' → '.join(plan.settings.variant_settings))
        return {'status':'preview','plan':plan.preview()}
    require(run_mode in {'train','resume'},'study mode preview/train/resume only')
    require(run_mode!='resume' or plan.action=='resume','resume mode requires a verified compatible saved study; no fresh training substituted')
    if notebook and getattr(plan,'reused_runs',None):
        for variant,run in plan.reused_runs.items():
            print(f"Reusing {variant} for evaluation only: {run.get('Drive_checkpoint') or run['checkpoint']} (matched case, source, capacity, optimizer/query configuration and initialization).")
    # A rerun of the same fresh cell automatically resumes/skips verified members.
    result=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='resume' if plan.action=='resume' else 'train')
    if plan.model=='C_GEO':
        from .data import prepare_data
        from .consistency import reference_floor
        data=prepare_data(Path(plan.root)/'learning_diagnostics'/'fixed_data',plan.certificate,experiment='c_physics_ablation',seed=plan.settings.seed,
            pde='wave',family='square',count=min(plan.settings.query_count,1024),cache_reference=True,grid_size=4)
        floor,_=reference_floor(data,count=32)
        path=Path(result['snapshot']).parent.parent/'reference-residual-floor.json';atomic_json(path,floor)
        result['reference_residual_floor']=floor
        # Save via the SAME suite artifact transaction: the next read-only suite
        # resume bundles this file. Also use the existing verified artifact copier
        # immediately so a disconnect cannot lose the residual floor.
        from ..artifact_io import write_artifact
        from ..artifact_drive import sync_verified
        fp=write_artifact(path.parent/'diagnostics','reference-residual-floor',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':plan.suite_id},
            {'floor':floor},required=('floor',),idempotent=True)
        result['reference_floor_sync']=sync_verified(fp,plan.persistence)
    if notebook:display_study(result,plan)
    # Reuse the original verified diagnostic artifact and Drive copy functions
    # for concise figures/floor produced after the training runner returns.
    directory=Path(result['snapshot']).parent.parent
    from ..artifact_io import write_artifact
    from ..artifact_drive import sync_verified
    files={str(p.relative_to(directory)):p.read_bytes().hex() for p in directory.rglob('*') if p.is_file() and p.suffix in {'.png','.json'} and not {'snapshots','diagnostics','continuations'}&set(p.relative_to(directory).parts) and p.name!='latest.json'}
    bundle=write_artifact(directory/'diagnostics',None,{'artifact_type':'diagnostic_bundle','architecture_id':plan.model,'run_id':plan.suite_id,'benchmark':'G1_smooth_v1'}, {'files':files},required=('files',),idempotent=True)
    result['study_diagnostics_sync']=sync_verified(bundle,plan.persistence)
    if result['study_diagnostics_sync']['Drive_status']=='failed':
        import warnings
        warnings.warn('STUDY DIAGNOSTICS DRIVE SYNC FAILED; verified local report/figures retained: '+str(result['study_diagnostics_sync']['error']),RuntimeWarning)
    return result


def display_study(result,plan):
    from IPython.display import display,Markdown,Image
    rows=result.get('summary',{}).get('rows',[])
    cols=('model','quantity','independent_MSE','physical_RMSE','sensor_MSE','weak','initial_consistency','boundary_consistency','clipping_fraction','elapsed_seconds','peak_allocated_CUDA_bytes')
    def fmt(value):return f'{value:.5g}' if isinstance(value,float) else str(value)
    display(Markdown('| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(fmt(r.get(k)) for k in cols)+' |' for r in rows)))
    if result.get('reference_residual_floor'):display(Markdown('**Certified reference residual floor:** `'+json.dumps(result['reference_residual_floor']['weak'])+'` (same weak definition, CPU float64).'))
    display(Markdown(f"Execution **{result['status']}** · local suite `{result['snapshot']}` · Drive **{result.get('suite_sync',{}).get('Drive_status')}**. Paired TRAIN diagnostics; no new-case generalization claim."))
    for path in result.get('summary',{}).get('figures',[]):display(Image(filename=path))
    # Pick at most TWO informative reconstructions after actual metrics exist.
    # Selection is descriptive TRAIN diagnostics, never an automatic winner.
    completed=[(v,r) for e in result['entries'].values() for v,r in e['runs'].items() if r.get('report')]
    selected=sorted(completed,key=lambda pair:sum(m['standardized_MSE'] for m in pair[1]['report']['metrics']['field'].values()))
    choices=[selected[0],selected[-1]] if len(selected)>1 else selected
    from .inspection import load_model
    from .data import prepare_data
    from .plotting import plots
    import matplotlib.pyplot as plt
    for variant,run in choices:
        path=run.get('Drive_checkpoint') or run['checkpoint'];model,meta=load_model(path,plan.certificate,device=plan.settings.device)
        c=meta['config'];data=prepare_data(Path(plan.root)/'learning_diagnostics'/'fixed_data',plan.certificate,experiment=c['experiment'],seed=c['root_seed'],pde=c['pdes'][0],family=c['training_families'][0],
            count=min(c['query_count'],1024),conditioning_h=c['conditioning_h'],cache_reference=True,grid_size=32,evaluation_times=(0.,.05,.1,.25,.5,.75,1.))
        destination=Path(result['snapshot']).parent.parent/'representative'/variant
        files=plots(model,data,destination,label=variant+' · descriptive TRAIN extremes')
        plt.close('all')
        for file in files:
            if file.endswith('reconstruction.png'):display(Image(filename=file))


def available_wave_paths(*results,explicit=None):
    paths=dict(explicit or {})
    for result in results:
        if not isinstance(result,dict):continue
        for name,entry in result.get('entries',{}).items():
            if name not in {'wave_single','c_physics_ablation'}:continue
            for variant,run in entry.get('runs',{}).items():
                report=run.get('report',{})
                if report.get('case',{}).get('family') and (name=='wave_single' or report.get('case',{}).get('fields',{}).get('v0')):
                    paths.setdefault(report.get('model','C_GEO')+'/'+variant,run.get('Drive_checkpoint') or run.get('checkpoint'))
    if not paths.get('C_GEO/FULL') and paths.get('C_GEO/FULL_01'):
        paths['C_GEO/FULL']=paths.pop('C_GEO/FULL_01')  # Explicit .1/.1/.1 full arm; original variant remains in report metadata.
    for label in ('A_GEO/FULL','C_GEO/FULL','C_GEO/FIELD_ONLY'):paths.setdefault(label,None)
    return paths


def display_consistency(report,root):
    from IPython.display import display,Markdown,Image
    import matplotlib.pyplot as plt
    rows=[]
    for label,r in report['models'].items():
        for group,m in r['pointwise']['groups'].items():
            if group not in {'all','t0','early','middle','late'}:continue
            rows.append(f"|{label}|{group}|{m['u']['standardized_RMS']:.5g}|{m['v']['standardized_RMS']:.5g}|{m['kinematic']['standardized_RMS']:.5g}|{m['dynamic']['standardized_RMS']:.5g}|{r['weak']['weak']:.5g}|{r['reference_weak']['weak']:.5g}|")
    display(Markdown('|Model|Time group|u error RMS|v error RMS|Kinematic RMS|Dynamic interior RMS|Weak MSE|Reference weak MSE|\n|---|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)))
    physical_rows=[];initial_rows=[]
    for label,r in report['models'].items():
        all_metrics=r['pointwise']['groups']['all']
        physical_rows.append(f"|{label}|{r.get('objective_variant')}|{all_metrics['kinematic']['physical_RMS']:.5g}|{all_metrics['dynamic']['physical_RMS']:.5g}|{r['weak']['boundary']:.5g}|{r['weak']['initial']:.5g}|")
        for name,metric in r['reconstruction']['auxiliary'].items():
            initial_rows.append(f"|{label}|{name}|{metric['standardized_MSE']:.5g}|{metric['physical_RMSE']:.5g}|")
    if physical_rows:display(Markdown('|Model|Saved variant|Kinematic physical RMS|Dynamic interior physical RMS|Boundary MSE|Initial consistency MSE|\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(physical_rows)))
    if initial_rows:display(Markdown('|Model|Initial auxiliary|Standardized MSE|Physical RMSE|\n|---|---|---:|---:|\n'+'\n'.join(initial_rows)))
    display(Markdown('**'+report['comparison']+'**. Strong residuals are sampled element-interior diagnostics; global interface terms are excluded. Reconstruction and physical consistency are separate.'))
    for label,message in report['missing'].items():print(label+': '+message)
    if not report['models']:return
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for label,r in report['models'].items():
        points=[(float(k[2:]),v) for k,v in r['pointwise']['groups'].items() if k.startswith('t=')];points.sort()
        for ax,key in zip(axs.ravel(),('u','v','kinematic','dynamic')):
            ax.plot([t for t,m in points],[m[key]['standardized_RMS'] for t,m in points],'-o',label=label)
            ax.plot([t for t,m in points],[r['reference_floor']['groups'][f't={t:g}'][key]['standardized_RMS'] for t,m in points],'--',alpha=.5,label=label+' reference floor' if key in {'kinematic','dynamic'} else None)
            ax.set(xlabel='physical time',ylabel=key+' standardized RMS',yscale='log')
    for ax in axs.ravel():ax.legend(fontsize=7)
    path=Path(root)/'wave-consistency.png';fig.savefig(path,dpi=130);plt.close(fig);display(Image(filename=str(path)))

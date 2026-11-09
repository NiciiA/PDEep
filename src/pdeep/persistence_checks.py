"""Bounded Stage7C orchestration: 8 versus 4+4 real updates per architecture.

Each interrupted phase and each inference reload is a distinct Python process.
The two-update end-to-end demonstration is reused inside the interrupted run;
there is no extra independent training suite or test-split evaluation.
"""
from dataclasses import asdict
from pathlib import Path
import json,subprocess,sys,time,uuid
import torch
from .core import require
from .artifact_io import atomic_json,read_manifest
from .artifact_tensors import tree_cpu
from .diagnostics import source_identity

PROTOCOL='stage7C-controlled-CPU-8-vs-4plus4-v2'


def _worker(root,directory,label,args):
    output=directory/(label+'.json');log=directory/(label+'.log')
    command=[sys.executable,'-m','pdeep.persistence_worker',*args,'--output',str(output)]
    with log.open('w') as stream:
        process=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    require(process.returncode==0,f'fresh-process {label} failed; see {log}')
    return json.loads(output.read_text())


def tree_difference(left,right):
    """Strict schema + actual max discrepancy; no tolerance inflation."""
    if isinstance(left,torch.Tensor):
        require(isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype,'resume tensor schema mismatch')
        return float((left.double()-right.double()).abs().max()) if left.numel() else 0.
    if isinstance(left,dict):
        require(isinstance(right,dict) and left.keys()==right.keys(),'resume mapping schema mismatch')
        return max((tree_difference(left[k],right[k]) for k in left),default=0.)
    if isinstance(left,(tuple,list)):
        require(isinstance(right,(tuple,list)) and len(left)==len(right),'resume sequence schema mismatch')
        return max((tree_difference(a,b) for a,b in zip(left,right)),default=0.)
    if isinstance(left,(int,float)) and not isinstance(left,bool):return abs(float(left)-float(right))
    require(left==right,'resume metadata/data-order/RNG marker differs');return 0.


def run_persistence_checks(root,*,fresh=False,progress=print):
    root=Path(root).resolve();source=source_identity(root);pointer=root/'reports/latest_persistence.json'
    if pointer.exists() and not fresh:
        prior=json.loads(pointer.read_text());record=json.loads((root/prior['report']).read_text())
        if record['source_identity']==source['sha256'] and record['protocol']==PROTOCOL:
            for manifest in record['verified_artifacts']:read_manifest(manifest)
            return record
        raise ValueError('persistence verification belongs to another source/configuration; explicitly request a fresh bounded verification')
    from .data import calibrate
    from .train_types import ResolvedTrainConfig
    from .inference_artifacts import export_inference,inference_examples,unpack_observation,unpack_query
    from .diagnostic_artifacts import save_diagnostic_bundle,capture_intermediates
    from .reference import canonical_spec
    started=time.monotonic();run_tag='stage7C-'+source['sha256'][:10]+'-'+uuid.uuid4().hex[:8]
    directory=root/'reports/persistence'/run_tag;directory.mkdir(parents=True,exist_ok=False)
    norm=calibrate();norm_path=norm.save(directory/'normalizers');rows={};artifacts=[]
    for architecture in ('A','C','B0','B1'):
        progress(f'{architecture}: controlled uninterrupted8, interrupted4 + fresh-process4; no research training')
        outputs={}
        for lane,stop in (('uninterrupted',8),('interrupted',4)):
            cfg=ResolvedTrainConfig(architecture,preset='bounded',updates=8,device='cpu',root_seed=norm.root_seed,source_id=source['sha256'],
                  run_tag=run_tag+'-'+lane,microbatch=1 if architecture=='C' else 2,accumulation=3 if architecture=='C' else 2,effective_batch=3,
                  query_count=8,validation_query_count=8,validation_cases_per_pde=1,validation_name='persistence-CPU-v2',
                  validation_every=2,checkpoint_every=2,scheduler='warmup_then_cosine',warmup_updates=2)
            config_path=directory/(architecture+'-'+lane+'-config.json');atomic_json(config_path,asdict(cfg))
            args=['train','--architecture',architecture,'--config',str(config_path),'--normalizers',str(norm_path),
                  '--run-root',str(root/'artifacts/persistence_runs'/run_tag/lane),'--stop',str(stop)]
            if lane=='interrupted':args+=['--workflow']
            outputs[lane]=_worker(root,directory,architecture+'-'+lane,args)
        outputs['resumed']=_worker(root,directory,architecture+'-resumed',
                                   ['resume','--architecture',architecture,'--manifest',outputs['interrupted']['checkpoint'],'--stop','8'])
        a=torch.load(outputs['uninterrupted']['snapshot'],map_location='cpu',weights_only=True)
        b=torch.load(outputs['resumed']['snapshot'],map_location='cpu',weights_only=True)
        discrepancies={name:tree_difference(a[name],b[name]) for name in ('model','optimizer','scheduler','predictions','trace','sampler','best_metric','best_step','successful','attempted','skipped','failed')}
        require(all(v==0 for v in discrepancies.values()),f'{architecture} deterministic CPU resume differs: {discrepancies}')
        require(a['successful']==b['successful']==8 and a['attempted']==b['attempted']==8 and a['failed']==b['failed']==a['skipped']==b['skipped']==0,'resume counters wrong')
        require(outputs['interrupted']['PID']!=outputs['resumed']['PID']!=outputs['uninterrupted']['PID'],'resume did not cross process boundary')
        workflow=json.loads((Path(outputs['interrupted']['run_directory'])/'end_to_end.json').read_text())
        independent=_worker(root,directory,architecture+'-independent-inference',
                             ['verify-inference','--architecture',architecture,'--manifest',workflow['inference_export']])
        current=read_manifest(outputs['resumed']['checkpoint']);checkpoint_dir=Path(outputs['resumed']['checkpoint']).parent
        last=json.loads((checkpoint_dir/'last.json').read_text());best=json.loads((checkpoint_dir/'best_validation.json').read_text())
        require(last['artifact_id']==current['artifact_id'],'last checkpoint pointer incorrect')
        best_record=read_manifest(checkpoint_dir/best['manifest']);require(best_record['model_selection']['is_best'],'best-validation pointer not attached to a best validation checkpoint')
        verified=[workflow['checkpoint'],workflow['inference_export'],workflow['diagnostic_bundle'],outputs['resumed']['checkpoint']]
        for path in verified:read_manifest(path)
        artifacts.extend(verified)
        rows[architecture]={'passed':True,'successful_updates_executed':16,'verification_device':'CPU float32; explicit CPU64 B bridge',
                            'end_to_end':workflow,'independent_inference':independent,'processes':outputs,'max_discrepancies':discrepancies,
                            'component_inventory':current['component_inventory'],'last_checkpoint':last,'best_validation_checkpoint':best,
                            'subsequent_case_and_observation_IDs_equal':True,'RNG_continuation_markers_equal':True,'source_identity':source['sha256']}
    example=inference_examples(norm)[-1];spec=canonical_spec('wave');obs=unpack_observation(example['observations']);q=unpack_query(example['queries'])
    d_export=export_inference(None,root/'artifacts/persistence_runs'/run_tag/'D',architecture='D',normalizers=norm,run_id=run_tag+'-D',
                              examples=[example],status='exact Gaussian; no training')
    d_diagnostic=save_diagnostic_bundle(root/'artifacts/persistence_runs'/run_tag/'D',architecture='D',normalizers=norm,run_id=run_tag+'-D',
                                       inputs=example,intermediates=capture_intermediates('D',None,spec,obs,q),packet={'track':'T1','posterior_mean_not_sample':True})
    d_reload=_worker(root,directory,'D-independent-inference',['verify-inference','--architecture','D','--manifest',str(d_export)])
    artifacts.extend([str(d_export),str(d_diagnostic)])
    report={'passed':True,'protocol':PROTOCOL,'run_id':run_tag,'source_identity':source['sha256'],'source':source,
            'architecture_order':['A','C','B0','B1','D'],'normalizer_identity':norm.identity,'neural_checks':rows,
            'D':{'export':str(d_export),'diagnostic':str(d_diagnostic),'independent_reload':d_reload},
            'verified_artifacts':artifacts,'actual_scientific_optimizer_updates_this_suite':64,
            'full_research_training_launched':False,'test_split_used':False,'elapsed_seconds':time.monotonic()-started,
            'Drive':{'actual_Drive_enabled':False,'real_Drive_tested':False,'reason':'no mounted Drive/credentials required; filesystem transaction tests are separate'},
            'device_limits':'strict same-host controlled CPU equality; accelerator reload tolerance/RNG support reported separately, no cross-hardware bitwise claim'}
    atomic_json(directory/'verification.json',report)
    atomic_json(pointer,{'report':str((directory/'verification.json').relative_to(root)),'source_identity':source['sha256'],'protocol':PROTOCOL})
    return report

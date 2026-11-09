"""Fresh-process persistence checks. Inference branch imports no training code."""
import argparse,json,sys,random
from pathlib import Path


def setup_cpu(*,strict_determinism=True):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(strict_determinism)


def verify_export(args):
    from .inference_artifacts import load_inference,verify_inference_examples
    from .artifact_io import read_manifest
    model=load_inference(args.manifest,expected_architecture=args.architecture,device=args.device,verify_examples=False)
    result=verify_inference_examples(model)
    forbidden=[n for n in sys.modules if n=='pdeep.training' or n.startswith('pdeep.train_') or n in {'pdeep.a_training','pdeep.b_training','pdeep.c_training','pdeep.b_artifacts'}]
    if forbidden:raise RuntimeError('inference reload imported training workflow: '+str(forbidden))
    result.update(fresh_process=True,PID=__import__('os').getpid(),training_modules_imported=forbidden,
                  artifact_id=read_manifest(args.manifest)['artifact_id'],manifest=str(Path(args.manifest).resolve()))
    import torch
    result['strict_deterministic_algorithms']=torch.are_deterministic_algorithms_enabled()
    result['reproducibility_scope']='same-host controlled CPU' if args.device=='cpu' else 'explicit accelerator inference tolerance; no deterministic scatter or bitwise guarantee'
    return result


def rng_marker(engine):
    import numpy as np,torch
    return {'python':random.random(),'numpy_global':float(np.random.random()),'torch_CPU':float(torch.rand(())),
            'numpy_registered':float(engine.rng_generators['verification_numpy'].random()),
            'torch_registered':float(torch.rand((),generator=engine.rng_generators['verification_torch']))}


def checkpoint_workflow(engine,root):
    """Two-update E2E evidence reuses the interrupted eight-update experiment."""
    from .artifact_io import read_manifest,atomic_json
    from .inference_artifacts import export_inference,inference_examples,unpack_observation,unpack_query
    from .diagnostic_artifacts import save_diagnostic_bundle,capture_intermediates
    from .reference import canonical_spec
    from .artifact_tensors import state_identity
    cp=Path(engine.state.last_checkpoint);record=read_manifest(cp)
    export=export_inference(engine.model,engine.run_dir/'inference',architecture=engine.config.architecture,
                            normalizers=engine.normalizers,run_id=engine.config.run_id,config=engine.config,successful=2,
                            parents=[{'artifact_id':record['artifact_id'],'path_hint':str(cp)}],status='bounded verification; two successful updates')
    example=inference_examples(engine.normalizers)[-1];spec=canonical_spec('wave')
    obs=unpack_observation(example['observations']);q=unpack_query(example['queries'])
    diagnostic=save_diagnostic_bundle(engine.run_dir/'diagnostics',architecture=engine.config.architecture,normalizers=engine.normalizers,
                                      run_id=engine.config.run_id,model=engine.model,config=engine.config,successful=2,
                                      inputs=example,intermediates=capture_intermediates(engine.config.architecture,engine.model,spec,obs,q),
                                      gradients={n:p.grad for n,p in engine.model.named_parameters() if p.grad is not None},
                                      packet={'capability':engine.capability,'validation':engine.latest_validation.metrics,'accepted_successful_updates':2},
                                      parents=[{'artifact_id':record['artifact_id'],'path_hint':str(cp)}])
    info={'architecture':engine.config.architecture,'run_id':engine.config.run_id,'successful':2,
          'checkpoint':str(cp),'checkpoint_artifact_id':record['artifact_id'],'inference_export':str(export),
          'diagnostic_bundle':str(diagnostic),'model_state_identity':state_identity(engine.model),'gradient_forward_checks':engine.capability}
    atomic_json(engine.run_dir/'end_to_end.json',info)
    return info


def train_worker(args):
    import numpy as np,torch
    from .train_types import ResolvedTrainConfig
    from .core import decode
    from .data import Normalizers
    from .training import TrainingEngine
    from .train_checkpoint import load_checkpoint
    from .artifact_io import append_event,atomic_json,read_manifest
    from .artifact_tensors import tree_cpu
    from .inference_artifacts import InferenceModel,inference_examples,unpack_observation,unpack_query
    from .reference import canonical_spec
    if args.action=='resume':
        engine=load_checkpoint(args.manifest,expected_architecture=args.architecture,probe=False,restore_rng=True)
    else:
        random.seed(771);np.random.seed(772);torch.manual_seed(773)
        config=decode(ResolvedTrainConfig,json.loads(Path(args.config).read_text()));norm=Normalizers.load(args.normalizers)
        engine=TrainingEngine(config,norm,args.run_root,probe=True);engine.state.stage='stage7C_bounded_resume_verification'
        engine.rng_generators={'verification_numpy':np.random.default_rng(774),'verification_torch':torch.Generator().manual_seed(775)}
    initial=engine.state.successful
    while engine.state.successful<args.stop:
        marker=rng_marker(engine);step=engine.update()
        append_event(engine.run_dir/'continuation_trace.jsonl',{'successful':step.successful,'markers':marker,
                     'identities':step.details['identities'],'loss':step.total,'unweighted':step.unweighted,'weighted':step.weighted,
                     'learning_rate':step.learning_rate,'next_learning_rate':step.next_learning_rate})
        if engine.state.successful%engine.config.validation_every==0:engine.validate()
        if engine.state.successful%engine.config.checkpoint_every==0:engine.save_checkpoint()
        if args.workflow and engine.state.successful==2:checkpoint_workflow(engine,args.run_root)
    wrapper=InferenceModel(engine.config.architecture,engine.model,{}, {},engine.config.device)
    modes=[(m,m.training) for m in engine.model.modules()];engine.model.eval();predictions={}
    try:
        for example in inference_examples(engine.normalizers):
            ctx=wrapper.prepare(canonical_spec(example['pde']),unpack_observation(example['observations']))
            predictions[example['pde']]=wrapper.predict(ctx,unpack_query(example['queries']),chunk=3)
    finally:
        for module,mode in modes:module.training=mode
    trace=[json.loads(line) for line in (engine.run_dir/'continuation_trace.jsonl').read_text().splitlines()]
    snapshot={'model':tree_cpu(engine.model.state_dict()),'optimizer':tree_cpu(engine.optimizer.state_dict()),'scheduler':engine.scheduler.state_dict(),
              'successful':engine.state.successful,'attempted':engine.state.attempted,'skipped':engine.state.skipped,'failed':engine.state.failed,
              'predictions':predictions,'trace':trace,'sampler':engine.sampler.state_dict(engine.state.sampler_cursor),
              'best_metric':engine.state.best_validation_metric,'best_step':engine.state.best_validation_step}
    target=engine.run_dir/f'verification-snapshot-{engine.state.successful:02d}.pt';torch.save(snapshot,target)
    output={'passed':True,'architecture':engine.config.architecture,'run_id':engine.config.run_id,'PID':__import__('os').getpid(),
            'started_at':initial,'stopped_at':engine.state.successful,'snapshot':str(target),'run_directory':str(engine.run_dir),
            'checkpoint':engine.state.last_checkpoint,'last_pointer':str(engine.run_dir/'checkpoints/last.json'),
            'best_pointer':str(engine.run_dir/'checkpoints/best_validation.json'),'RNG_restore_limitations':getattr(engine,'rng_restore_limitations',[])}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify-inference','train','resume'])
    parser.add_argument('--architecture',required=True,choices=['A','C','B0','B1','D'])
    parser.add_argument('--manifest');parser.add_argument('--config');parser.add_argument('--normalizers');parser.add_argument('--run-root')
    parser.add_argument('--stop',type=int,default=8);parser.add_argument('--workflow',action='store_true')
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    # MPS scatter_add has no deterministic implementation. This is an explicit
    # accelerator inference policy, not a CPU-resume tolerance or model change.
    setup_cpu(strict_determinism=args.action!='verify-inference' or args.device=='cpu')
    result=verify_export(args) if args.action=='verify-inference' else train_worker(args)
    from .artifact_io import atomic_json
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'cases','trace'}},indent=2))

if __name__=='__main__':main()

"""Bounded fresh-kernel verification of the three routine notebooks.

Saved defaults never train. One copied training notebook is interrupted after
successful update1 and resumes to its saved two-update budget (B1 only). An
explicit partial-checkpoint evaluation copy compares existing B0/B1 exports + D.
No scientific source, original experiment or saved launch default is modified.
"""
import argparse,copy,json,os,sys,tempfile,time,uuid
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from pdeep.diagnostics import source_identity


def execute(notebook,root,tmp,*,timeout=90):
    manager=AsyncKernelManager(kernel_name='pdeep-routine',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(tmp/'kernels')]),
                               connection_file=str(tmp/(uuid.uuid4().hex+'.json')))
    client=NotebookClient(notebook,km=manager,timeout=timeout,kernel_name='pdeep-routine',resources={'metadata':{'path':str(root/'notebooks')}})
    env=dict(os.environ);env['IPYTHONDIR']=str(tmp/'ipython');env['MPLCONFIGDIR']=str(tmp/'matplotlib')
    start=time.monotonic()
    try:client.execute(env=env,cwd=str(root/'notebooks'))
    finally:
        import asyncio
        if manager.has_kernel:asyncio.run(manager.shutdown_kernel(now=True))
        asyncio.run(manager.cleanup_resources())
    return {'passed':True,'cells':len(notebook.cells),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),'seconds':time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    root=args.root.resolve();run='routine-'+source_identity(root)['sha256'][:12]+'-'+uuid.uuid4().hex[:6]
    directory=root/'reports/notebook_workflows'/run;directory.mkdir(parents=True)
    reports={};start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='pdeep-routine-kernel-') as value:
        tmp=Path(value);kernel=tmp/'kernels/pdeep-routine';kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'PDEep routine checks','language':'python'}))
        for name in ('01_explore.ipynb','02_train.ipynb','03_evaluate.ipynb'):
            print('Fresh safe defaults:',name,flush=True)
            path=root/'notebooks'/name;n=nbformat.read(path,as_version=4)
            reports[name]=execute(n,root,tmp);nbformat.write(n,path)
        # Verification copy: production saved defaults above remain preview.
        n=nbformat.read(root/'notebooks/02_train.ipynb',as_version=4)
        for c in n.cells:
            if c.cell_type=='code' and 'advanced' in c.metadata.get('tags',[]):
                c.source+='''\n# Bounded verification-only override (not written to the saved notebook).
MODELS=("B1",); DEVICE="cpu"; PRESET="smoke"; UPDATES=2
SAVE_EVERY=1; VALIDATE_EVERY=1; RUN_MODE="train"
ADVANCED={"microbatch":1,"accumulation":1,"effective_batch":1,"query_count":8,"validation_cases_per_pde":1,"validation_query_count":8}
RUN_NAME='''+repr(run)+'''
from pdeep.workflow_plotting import TrainingMonitor
class InterruptAfterFirstCheckpoint(TrainingMonitor):
    def __init__(self):
        super().__init__(notebook=True);self.did_interrupt=False
    def update(self,config,state,step=None,validation=None):
        result=super().update(config,state,step,validation)
        if not self.did_interrupt and state.successful==1 and state.last_checkpoint and state.phase=="ready":
            self.did_interrupt=True
            raise KeyboardInterrupt("bounded verification interruption at completed checkpoint1")
        return result
BOUNDED_MONITOR=InterruptAfterFirstCheckpoint()
'''
            if c.cell_type=='code' and 'launch' in c.metadata.get('tags',[]):
                c.source=c.source.replace('models=MODELS_READY,notebook=True)','models=MODELS_READY,notebook=True,monitor=BOUNDED_MONITOR)')
        n.cells.append(nbformat.v4.new_code_cell('''from pdeep.train_checkpoint import load_checkpoint
from pdeep.artifact_tensors import tree_identity,state_identity
from pdeep.core import digest
from pdeep.artifact_io import read_manifest
first=TRAINING_RESULTS["B1"]; engine=first["engine"]
assert first["interrupted"] and engine.state.successful==1
checkpoint=first["last_checkpoint"]
restored=load_checkpoint(checkpoint,expected_architecture="B1",restore_rng=False,probe=False)
assert state_identity(restored.model)==state_identity(engine.model)
assert digest(tree_identity(restored.optimizer.state_dict()))==digest(tree_identity(engine.optimizer.state_dict()))
assert restored.scheduler.state_dict()==engine.scheduler.state_dict()
_,expected_ids=engine.sampler.batch(engine.state.sampler_cursor)
resume_plan=prepare_training(CONTEXT,models=("B1",),pde_families=PDE_FAMILIES,device="cpu",action="resume",resume_checkpoint=checkpoint)
assert resume_plan.preview["entries"][0]["completed_successful_updates"]==1
assert resume_plan.preview["entries"][0]["remaining_successful_updates"]==1
resumed=launch_training(resume_plan,run_mode="resume",notebook=True)
result=resumed["B1"]; e=result["engine"]
assert e.state.successful==2 and e.state.attempted==2 and e.state.failed==e.state.skipped==0
assert result["new_successful_updates"]==1 and e.scheduler.completed==2
trace=[json.loads(line) for line in (e.run_dir/"steps.jsonl").read_text().splitlines()]
assert trace[1]["details"]["identities"]==json.loads(json.dumps(expected_ids))
assert BOUNDED_MONITOR.did_interrupt and BOUNDED_MONITOR.validation_seen and len(BOUNDED_MONITOR.events)>4
assert not BOUNDED_MONITOR.plot_errors
print("Live events",len(BOUNDED_MONITOR.events),"| original validation callbacks",len(BOUNDED_MONITOR.validation_seen))
info={"passed":True,"architecture":"B1","successful_updates":e.state.successful,"new_resume_updates":1,
      "checkpoint":result["last_checkpoint"],"inference_export":result["inference_export"],
      "first_checkpoint":checkpoint,"optimizer_and_model_reload_equal":True,"next_case_observation_query_IDs_equal":True,
      "monitor_live_events_before_completion":len(BOUNDED_MONITOR.events),"live_validation_plots_passed":not BOUNDED_MONITOR.plot_errors}
Path('''+repr(str(directory/'smoke_resume.json'))+''').write_text(json.dumps(info,indent=2)+"\\n")
''',metadata={'tags':['implemented','verification-only']}))
        print('Explicit B1 smoke: interrupt after1, resume to2 (two total updates)',flush=True)
        reports['explicit_smoke_resume']=execute(n,root,tmp);nbformat.write(n,directory/'bounded_train_executed.ipynb')
        smoke=json.loads((directory/'smoke_resume.json').read_text())
        import subprocess
        command=[sys.executable,'-m','pdeep.persistence_worker','verify-inference','--architecture','B1',
                 '--manifest',smoke['inference_export'],'--output',str(directory/'independent_smoke_reload.json')]
        cp=subprocess.run(command,capture_output=True,text=True,timeout=90)
        (directory/'independent_smoke_reload.log').write_text(cp.stdout+cp.stderr)
        if cp.returncode:raise RuntimeError('independent smoke export reload failed')
        # Explicitly opt into actual bounded execution artifacts, not fake trained models.
        saved=json.loads((root/'reports/latest_persistence.json').read_text());previous=json.loads((root/saved['report']).read_text())
        selected={name:previous['neural_checks'][name]['end_to_end']['inference_export'] for name in ('B0','B1')}
        n=nbformat.read(root/'notebooks/03_evaluate.ipynb',as_version=4)
        for c in n.cells:
            if c.cell_type=='code' and 'configuration' in c.metadata.get('tags',[]):
                c.source+='\nCHECKPOINTS='+repr(selected)+'\nALLOW_PARTIAL=True\nDEVICE="cpu"\n'
        n.cells.append(nbformat.v4.new_code_cell('''assert set(RESULT["scientific"]["methods"])=={"B0","B1","D"}
assert RESULT["scientific"]["optimizer_steps"]==0
for name in ("B0","B1"):
    meta=RESULT["scientific"]["methods"][name]
    assert meta["training_status"]=="bounded" and meta["training_selection"]["successful_updates"]==2
assert RESULT["scientific"]["methods"]["D"]["training_status"]=="exact; no training"
Path('''+repr(str(directory/'comparison.json'))+''').write_text(json.dumps({"passed":True,"directory":RESULT["directory"],"status":"explicit bounded2-update B0/B1 + exact D; not research models"},indent=2)+"\\n")
''',metadata={'tags':['implemented','verification-only']}))
        print('Explicit bounded-checkpoint comparison: B0/B1 + D, zero updates',flush=True)
        reports['B0_B1_D_comparison']=execute(n,root,tmp);nbformat.write(n,directory/'bounded_evaluation_executed.ipynb')
    output={'passed':True,'run_id':run,'source_identity':source_identity(root)['sha256'],'notebooks':reports,
            'smoke_resume':smoke,'independent_smoke_reload':json.loads((directory/'independent_smoke_reload.json').read_text()),
            'comparison':json.loads((directory/'comparison.json').read_text()),'scientific_optimizer_updates':2,
            'research_training':False,'test_split_used':False,'Colab_CUDA_Drive_executed':False,'seconds':time.monotonic()-start}
    (directory/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
    (root/'reports/latest_notebook_workflows.json').write_text(json.dumps({'report':str((directory/'verification.json').relative_to(root))},indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in {'independent_smoke_reload','notebooks'}},indent=2))

if __name__=='__main__':main()

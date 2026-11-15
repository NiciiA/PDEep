"""Fresh-kernel preview checks; forbid optimizers, reference solves and local Drive."""
from pathlib import Path
import json,os,sys,tempfile,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
root=Path(__file__).resolve().parents[1];reports={}
with tempfile.TemporaryDirectory(prefix='g1-mode-notebooks-') as value:
    tmp=Path(value);spec=tmp/'kernels/pdeep-modes';spec.mkdir(parents=True);(spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'G1 modes safe checks','language':'python'}))
    for name in ('04_g1_unified.ipynb','05_g1_playground.ipynb'):
        n=nbformat.read(root/'notebooks'/name,4);n.cells.insert(0,nbformat.v4.new_code_cell('''import sys,torch
from pdeep.unified import dataset
from pdeep.unified_bayes.operators import ForwardOperators
from pdeep.smooth.reference import Reference

def forbidden(*a,**kw):
    raise AssertionError("Safe notebook attempted training, reference preparation or Drive")
torch.optim.AdamW.__init__ = forbidden
dataset.prepare_dataset = forbidden
ForwardOperators.prepare = forbidden
Reference.__init__ = forbidden
class NoDrive:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith("google.colab.drive"):
            raise AssertionError("Local preview imported Drive")
sys.meta_path.insert(0,NoDrive())
'''))
        km=AsyncKernelManager(kernel_name='pdeep-modes',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(tmp/'kernels')]),connection_file=str(tmp/(name+'.json')));start=time.perf_counter()
        env=dict(os.environ,IPYTHONDIR=str(tmp/'ipython'),MPLCONFIGDIR=str(tmp/'matplotlib'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
        try:NotebookClient(n,km=km,timeout=120,resources={'metadata':{'path':str(root)}}).execute(env=env,cwd=str(root))
        finally:
            import asyncio
            if km.has_kernel:asyncio.run(km.shutdown_kernel(now=True))
            asyncio.run(km.cleanup_resources())
        out=root/'reports/g1_modes';out.mkdir(parents=True,exist_ok=True);nbformat.write(n,out/(name+'.executed.ipynb'));reports[name]={'passed':True,'seconds':time.perf_counter()-start,'cells':len(n.cells)-1,'optimizer_updates':0,'reference_solves':0,'Drive_imports':0};print(name,reports[name],flush=True)
(root/'reports/g1_modes/notebooks.json').write_text(json.dumps(reports,indent=2)+'\n')

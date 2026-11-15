"""Fresh-kernel safe Run All. Guards forbid optimizers, PDE preparation and Drive."""
from pathlib import Path
import json,os,sys,tempfile,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

root=Path(__file__).resolve().parents[1];n=nbformat.read(root/'notebooks/04_g1_unified.ipynb',4)
guard=nbformat.v4.new_code_cell('''import sys,torch
from pdeep.unified_bayes import operators
from pdeep.unified import dataset

def expensive(*a,**kw):
    raise AssertionError("Safe Run All attempted optimization, reference preparation or Drive")
torch.optim.AdamW.__init__ = expensive
operators.ForwardOperators.prepare = expensive
dataset.prepare_dataset = expensive
from pdeep.smooth.reference import Reference
Reference.__init__ = expensive
class ForbiddenDriveImport:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith("google.colab.drive"):
            raise AssertionError("Local preview imported Drive")
sys.meta_path.insert(0,ForbiddenDriveImport())
''')
n.cells.insert(0,guard);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='g1-bayes-safe-') as value:
    tmp=Path(value);spec=tmp/'kernels/pdeep-g1';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'G1 safe test','language':'python'}))
    km=AsyncKernelManager(kernel_name='pdeep-g1',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(tmp/'kernels')]),connection_file=str(tmp/'connection.json'))
    env=dict(os.environ,IPYTHONDIR=str(tmp/'ipython'),MPLCONFIGDIR=str(tmp/'matplotlib'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    try:NotebookClient(n,km=km,timeout=120,resources={'metadata':{'path':str(root)}}).execute(env=env,cwd=str(root))
    finally:
        import asyncio
        if km.has_kernel:asyncio.run(km.shutdown_kernel(now=True))
        asyncio.run(km.cleanup_resources())
    out=root/'reports/g1_bayesian';out.mkdir(parents=True,exist_ok=True);nbformat.write(n,out/'safe-defaults.executed.ipynb')
    report={'passed':True,'seconds':time.perf_counter()-start,'cells':len(n.cells)-1,'guards':'no optimizer construction, no PDE reference construction/preparation, no local Drive import','notebook_modified':False}
    (out/'notebook_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

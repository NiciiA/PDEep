"""Actual fresh Jupyter execution, forbidding all engine/optimizer calls."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/colab_repair'
guard='''
import numpy  # exercise a preloaded binary; no replacement is allowed
import torch
ENGINE_CALLS=[]
OPTIMIZER_CALLS=[]
def forbidden_engine(*a,**k):
    ENGINE_CALLS.append(1);raise AssertionError("Run All tried to construct/launch a training engine")
def forbidden_optimizer(*a,**k):
    OPTIMIZER_CALLS.append(1);raise AssertionError("Run All tried an optimizer update")
for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):cls.step=forbidden_optimizer
from pdeep.training import TrainingEngine
from pdeep.smooth.training import Engine
from pdeep.geometric.training import Engine as RoughEngine
for cls in (TrainingEngine,Engine,RoughEngine):
    for method in ("__init__","run","update"):setattr(cls,method,forbidden_engine)
'''
for name in ('02_train','03_evaluate'):
 nb=nbformat.read(ROOT/'notebooks'/f'{name}.ipynb',4)
 if name=='02_train':
  for c in nb.cells:
   if c.cell_type=='code':c.source=c.source.replace('RUN_MODE = "preview"','RUN_MODE = "train"').replace('DEVICE = "auto"','DEVICE = "cpu"')
 nb.cells.insert(0,nbformat.v4.new_code_cell(guard))
 nb.cells.append(nbformat.v4.new_code_cell('assert not ENGINE_CALLS and not OPTIMIZER_CALLS\nprint("PASSED fresh kernel: zero engine/optimizer calls")'))
 NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
 nbformat.write(nb,OUT/(name+'.executed.ipynb'))
result={'passed':True,'actual_fresh_Jupyter_kernels':2,'02_saved_mode':'train','engine_calls':0,'optimizer_calls':0,'preloaded_numpy_retained':True,'hosted_Colab_CUDA_Drive':'not tested'}
(OUT/'notebook-safety.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

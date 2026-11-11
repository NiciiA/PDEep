"""Fresh kernel with saved train mode; every training entry point is forbidden.

Never simulate a button click. Safe Run All must complete with zero calls.
"""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/smooth_finalization'
nb=nbformat.read(ROOT/'notebooks/02_train.ipynb',as_version=4)
for c in nb.cells:
 if c.cell_type=='code':c.source=c.source.replace('RUN_MODE = "preview"','RUN_MODE = "train"').replace('DEVICE = "auto"','DEVICE = "cpu"')
guard=r'''
import torch
OPTIMIZER_CALLS=[]
TRAINING_CALLS=[]
def _forbid_optimizer(*args,**kwargs):
    OPTIMIZER_CALLS.append(1)
    raise AssertionError("Fresh-kernel verification forbids optimizer updates")
def _forbid_training(*args,**kwargs):
    TRAINING_CALLS.append(1)
    raise AssertionError("Run All invoked a training engine")
for cls in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):cls.step=_forbid_optimizer
from pdeep.training import TrainingEngine
from pdeep.geometric.training import Engine as RoughEngine
from pdeep.smooth.training import Engine as SmoothEngine
for cls in (TrainingEngine,RoughEngine,SmoothEngine):
    for name in ("__init__","run","update"):
        setattr(cls,name,_forbid_training)
'''
nb.cells.insert(0,nbformat.v4.new_code_cell(guard))
nb.cells.append(nbformat.v4.new_code_cell('''assert not OPTIMIZER_CALLS and not TRAINING_CALLS
assert not RESULTS and RUN_MODE=="train"
assert LAUNCH_BUTTON.enabled
print("FRESH KERNEL PASSED: saved train mode; zero engine calls, zero optimizer calls, no button click.")
'''))
client=NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
client.execute();nbformat.write(nb,OUT/'02_train-saved-train-mode.executed.ipynb')
nb_eval=nbformat.read(ROOT/'notebooks/03_evaluate.ipynb',as_version=4)
nb_eval.cells.insert(0,nbformat.v4.new_code_cell(guard))
nb_eval.cells.append(nbformat.v4.new_code_cell('assert not OPTIMIZER_CALLS and not TRAINING_CALLS\nassert not RUN_SMOOTH_EVALUATION and not ENABLE_T1_EVALUATION\nprint("03 default fresh Run All: zero engine/optimizer calls.")'))
NotebookClient(nb_eval,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(nb_eval,OUT/'03_evaluate-default.executed.ipynb')
result={'passed':True,'kernel':'fresh real Jupyter python3','saved_mode':'train','engine_calls':0,'optimizer_calls':0,'button_clicks':0,'default_preview_also_safe':True,'03_default_Run_All_safe':True,'scope':'actual local notebook execution; no CUDA/Drive claim'}
(OUT/'notebook-safety.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

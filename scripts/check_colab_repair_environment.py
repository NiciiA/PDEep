"""Fresh/preloaded numerical-stack verification; no GPU/Colab claim or training."""
from pathlib import Path
import argparse,json,runpy,sys
parser=argparse.ArgumentParser();parser.add_argument('--preload',action='store_true');args=parser.parse_args()
if args.preload:import numpy
before=sys.modules.get('numpy');helper=runpy.run_path('src/pdeep/smooth/bootstrap.py')
def forbidden(*a,**k):raise AssertionError('an existing dependency was replaced')
first=helper['prepare_reference_environment'](Path('.'),is_colab=True,runner=forbidden)
second=helper['prepare_reference_environment'](Path('.'),is_colab=True,runner=forbidden)
import numpy,scipy,torch
from pdeep.smooth.qualification import ensure_compatible_certificate
from pdeep.smooth.workflows import prepare
record,path=ensure_compatible_certificate()
plan=prepare(models=('A_GEO','C_GEO'),experiment='tiny_overfit',updates=100,device='cpu',certificate_path=path)
assert not first['installed'] and not second['installed'] and not first['restart_required']
assert before is None or numpy is before
assert all(c.updates==100 for c in plan.configs)
result={'passed':True,'preloaded_numpy':args.preload,'same_loaded_numpy':before is None or numpy is before,'executable':sys.executable,'numpy':numpy.__version__,'scipy':scipy.__version__,'certificate':record['certification_id'],'models':[c.architecture for c in plan.configs],'requested_tiny_budget':100,'pip_calls':0,'optimizer_updates':0,'scope':'actual local isolated dependency profile, not hosted Colab/CUDA verification'}
print('RESULT_JSON '+json.dumps(result))

"""Run numerical/UI/architecture regressions with ALL optimizer steps forbidden.

Tests whose purpose is actually updating optimizers (legacy training engine and
optimizer-checkpoint arithmetic) are excluded. No A_GEO/C_GEO training is allowed.
"""
import sys
import torch,pytest
calls=[]
def forbidden(*args,**kwargs):
    calls.append('optimizer');raise AssertionError('G3 verification forbids neural optimizer updates')
for optimizer in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):optimizer.step=forbidden
status=pytest.main(['-q','--disable-warnings','--ignore=tests/test_training_engine.py','--ignore=tests/test_artifacts.py'])
print('G3 verification optimizer calls:',len(calls))
sys.exit(status if not calls else 1)

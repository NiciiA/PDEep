"""The original matched validation metric, bound to exact prepared draws.

Fixed query/noise/mask recipes are independent of model and checkpoint number.
No test sweep, reference solver or new validation sampling law is introduced.
"""
import types
from ..core import require
from ..unified.data import Provider as BaseProvider
from ..unified.config import SCHEMA as DATA_SCHEMA
from ..unified_modes import evaluation as original
from .coordinates import prepared_design

def _bind(function,**bindings):
    result=types.FunctionType(function.__code__,{**function.__globals__,**bindings},function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__;return result

class Provider(BaseProvider):
    def token_set(self,c,count,*,observation=False,set_id=0,coordinates=None,noise_multiplier=1.):
        require(c.split=='validation','training validation may only consume validation cases')
        if coordinates is None:coordinates=prepared_design(self.dataset,c,count,observation=observation,set_id=set_id)
        return super().token_set(c,count,observation=observation,set_id=set_id,coordinates=coordinates,noise_multiplier=noise_multiplier)

def evaluate(model,dataset,**kwargs):
    require(kwargs.get('split','validation')=='validation' and not kwargs.get('full',False) and not kwargs.get('grid_size') and not kwargs.get('diagnostics',False) and kwargs.get('operators') is None,'training validation uses its fixed prepared validation protocol only')
    require(kwargs.get('query_count',512)==dataset.config.stream_validation_queries,'validation query count differs from prepared schedule')
    def observations(c,n,seed):
        return prepared_design(dataset,c,n,observation=True,set_id=(DATA_SCHEMA,'fixed-evaluation',seed))
    view=_bind(original.information_view,Provider=Provider,observation_design=observations)
    return _bind(original.evaluate,Provider=Provider,information_view=view)(model,dataset,**kwargs)

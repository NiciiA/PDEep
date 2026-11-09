"""Full runtime RNG snapshots plus explicit stateless project-stream recipes."""
import random
import numpy as np
import torch
from .core import require


def _numpy_pack(value):
    if isinstance(value,np.ndarray):return {'numpy_array':value.tolist(),'dtype':str(value.dtype)}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_numpy_pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_numpy_pack(v) for v in value]
    return value


def _numpy_unpack(value):
    if isinstance(value,dict) and set(value)=={'numpy_array','dtype'}:
        require(value['dtype'] in {'uint32','uint64','int64','float64'},'unsupported NumPy RNG state dtype')
        return np.array(value['numpy_array'],dtype=value['dtype'])
    if isinstance(value,dict):return {k:_numpy_unpack(v) for k,v in value.items()}
    if isinstance(value,list):return [_numpy_unpack(v) for v in value]
    return value


def capture_rng(generators=None):
    legacy=np.random.get_state()
    state={'python_random':random.getstate(),'numpy_global':(legacy[0],legacy[1].tolist(),legacy[2],legacy[3],legacy[4]),
           'torch_CPU':torch.random.get_rng_state().clone(),'torch_CUDA':[], 'torch_MPS':None,
           'device_limitations':{},'registered':{}}
    if torch.cuda.is_available():state['torch_CUDA']=[s.cpu() for s in torch.cuda.get_rng_state_all()]
    else:state['device_limitations']['CUDA']='unavailable; no state captured'
    if torch.backends.mps.is_available() and hasattr(torch.mps,'get_rng_state'):
        try:state['torch_MPS']=torch.mps.get_rng_state().cpu().clone()
        except (RuntimeError,NotImplementedError) as error:state['device_limitations']['MPS']=str(error)
    else:state['device_limitations']['MPS']='unavailable or get_rng_state unsupported'
    for name,generator in (generators or {}).items():
        require(type(name) is str and name,'named RNG required')
        if isinstance(generator,np.random.Generator):
            state['registered'][name]={'kind':'numpy','bit_generator':type(generator.bit_generator).__name__,'state':_numpy_pack(generator.bit_generator.state)}
        elif isinstance(generator,torch.Generator):
            state['registered'][name]={'kind':'torch','device':str(generator.device),'state':generator.get_state().cpu().clone()}
        else:raise ValueError('registered RNG must be NumPy Generator or torch.Generator')
    return state


def restore_rng(state, *, target_device='cpu'):
    """Restore after construction/loading. No exact cross-device claim is made."""
    random.setstate(state['python_random']);n=state['numpy_global']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.random.set_rng_state(state['torch_CPU'])
    limitations=[];target=torch.device(target_device).type
    if state['torch_CUDA']:
        if torch.cuda.is_available() and len(state['torch_CUDA'])==torch.cuda.device_count():torch.cuda.set_rng_state_all(state['torch_CUDA'])
        else:
            require(target!='cuda','saved CUDA RNG/device count unavailable for exact resume');limitations.append('unused CUDA state not restored on this host')
    if state['torch_MPS'] is not None:
        if torch.backends.mps.is_available() and hasattr(torch.mps,'set_rng_state'):torch.mps.set_rng_state(state['torch_MPS'])
        else:
            require(target!='mps','saved MPS RNG cannot be restored on this host');limitations.append('unused MPS state not restored on this host')
    elif target=='mps':limitations.append('MPS RNG capture unsupported; exact stochastic-device continuation cannot be claimed')
    registered={}
    bit_generators={name:getattr(np.random,name) for name in ('PCG64','PCG64DXSM','MT19937','Philox','SFC64')}
    for name,entry in state['registered'].items():
        if entry['kind']=='numpy':
            require(entry['bit_generator'] in bit_generators,'unsupported NumPy RNG algorithm')
            bit=bit_generators[entry['bit_generator']]();bit.state=_numpy_unpack(entry['state']);registered[name]=np.random.Generator(bit)
        else:
            require(entry['kind']=='torch' and torch.device(entry['device']).type in {'cpu','cuda'},'unsupported registered torch generator')
            generator=torch.Generator(device=entry['device']);generator.set_state(entry['state']);registered[name]=generator
    return registered,limitations

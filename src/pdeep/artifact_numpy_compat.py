"""Narrow numeric-scalar compatibility for verified project legacy artifacts.

Always weights_only=True. No ndarray reconstruction, arbitrary dtype, dynamic
imports from pickle, global add_safe_globals or unsafe pickle fallback. The
caller verifies manifest identity AND payload SHA/size before reaching here;
recognized project schema and approved source are additionally required before
retry. Scoped aliases handle NumPy1/2 scalar paths and dynamic numeric dtypes.
"""
import pickle
from threading import RLock
from types import SimpleNamespace
import numpy as np
import torch
from .core import require,ValidationError

_NUMERIC=('bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64')
_DTYPE_TYPES={type(np.dtype(name)) for name in _NUMERIC}
_GLOBALS={'numpy.dtype','numpy.core.multiarray.scalar','numpy._core.multiarray.scalar'}
_LOAD_LOCK=RLock()


def _checked_dtype(*args,**kwargs):
    require(args and type(args[0]) is str and not kwargs,'legacy compatibility accepts only simple numeric dtype codes')
    try:dtype=np.dtype(*args)
    except (TypeError,ValueError) as error:
        raise ValidationError('unsupported legacy NumPy dtype') from error
    require(dtype.name in _NUMERIC and dtype.fields is None and dtype.subdtype is None and not dtype.hasobject,'unsupported legacy NumPy dtype')
    return dtype


def _checked_scalar(dtype,buffer):
    require(type(dtype) in _DTYPE_TYPES and dtype.name in _NUMERIC and not dtype.hasobject and dtype.fields is None and dtype.subdtype is None,'unsupported legacy NumPy scalar dtype')
    require(type(buffer) is bytes and len(buffer)==dtype.itemsize,'legacy numeric scalar requires exactly one fixed-size value')
    # The callable is installed code, never imported using a checkpoint string.
    core=getattr(np,'_core',None)
    if core is None:core=np.core
    return core.multiarray.scalar(dtype,buffer)


def _approved_source(source,config=None):
    from .learning_diagnostics.engine import diagnostic_source,approved_diagnostic
    if source==diagnostic_source():return True
    d={'query_revision':'legacy-fixed-v1','query_mode':'fixed','amp_mode':'off',
       'physics_weights':None,'physics_schedule':'constant','gradient_diagnostics_every':0,**(config or {}),'diagnostic_source_id':source}
    return approved_diagnostic(SimpleNamespace(**d))


def _check_project(record,payload=None):
    """Hashes alone are integrity, not source trust; retain the project gates."""
    from .smooth.compatibility import approved_predecessor
    if record.get('benchmark')=='G1_smooth_v1' and record.get('scientific_schema')=='G1_smooth-fixed-TRAIN-learning-diagnostics-v1':
        require(record.get('architecture_id') in {'A_GEO','C_GEO'} and record.get('artifact_type') in {'training_checkpoint','inference_export','diagnostic_bundle'},'legacy NumPy compatibility requires a recognized project artifact')
        config=record.get('config',{})
        require(record.get('implementation_identity') is not None and approved_predecessor(record['implementation_identity']),'legacy checkpoint implementation/source incompatible')
        source=record.get('diagnostic_source_id')
        require(source is not None and config.get('diagnostic_source_id',source)==source and _approved_source(source,config),'legacy checkpoint diagnostic source incompatible')
        if config.get('source_id'):require(approved_predecessor(config['source_id']),'legacy checkpoint configuration source incompatible')
        return
    if record.get('suite_schema')=='G1-smooth-diagnostic-suite-v1' and record.get('artifact_type')=='diagnostic_bundle':
        require(record.get('architecture_id') in {'A_GEO','C_GEO'},'unrecognized legacy suite architecture')
        if payload is not None:
            state=payload.get('suite_state',{})
            require(state.get('suite_id')==record.get('run_id') and state.get('configuration_hash')==record.get('configuration_hash'),'legacy suite identity mismatch')
            require(_approved_source(state.get('source')),'legacy suite source incompatible')
        return
    raise ValueError('NumPy scalar compatibility is restricted to verified, source-compatible Phase G project artifacts')


def load_verified_payload(path,record):
    """Called only after read_manifest's file verification; scope lasts one load."""
    # PyTorch's context temporarily registers globals process-wide. Serialize
    # project reads so another read cannot inherit that compatibility scope.
    with _LOAD_LOCK:return _load_verified_payload(path,record)


def _load_verified_payload(path,record):
    try:return torch.load(path,map_location='cpu',weights_only=True)
    except pickle.UnpicklingError as error:
        # Never respond to an arbitrary unpickling error by expanding trust.
        if 'numpy' not in str(error).lower() and 'dtype' not in str(error).lower():raise
        _check_project(record)
        inspect=getattr(torch.serialization,'get_unsafe_globals_in_checkpoint',None)
        require(inspect is not None,'legacy safe compatibility requires a PyTorch release with checkpoint-global inspection; upgrade PyTorch, never disable weights_only')
        unsafe=set(inspect(path))
        require(unsafe<=_GLOBALS,'unsupported globals in legacy checkpoint; only numeric NumPy scalar metadata is permitted: '+str(sorted(unsafe-_GLOBALS)))
        allowed=[(_checked_dtype,'numpy.dtype'),(_checked_scalar,'numpy.core.multiarray.scalar'),
                 (_checked_scalar,'numpy._core.multiarray.scalar'),*_DTYPE_TYPES]
        # Do not remove caller-owned safe globals when the temporary scope exits.
        existing=torch.serialization.get_safe_globals();allowed=[item for item in allowed if item not in existing]
        with torch.serialization.safe_globals(allowed):
            payload=torch.load(path,map_location='cpu',weights_only=True)
        _check_project(record,payload)
        return payload

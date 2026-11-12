"""Explicit controls for the existing diagnostic adapter; no training loop."""
from dataclasses import replace
from ..core import require
from ..geometric.physics import WeakConfig

C_VARIANTS={
    'FIELD_ONLY':dict(physics_weights=(0.,0.,0.)),
    'WEAK_001':dict(physics_weights=(.001,0.,0.)),
    'ALL_001':dict(physics_weights=(.001,.001,.001)),
    'ALL_01':dict(physics_weights=(.01,.01,.01)),
    'FULL_01':dict(physics_weights=(.1,.1,.1)),
}
STAGED=dict(physics_weights=(.001,.001,.001),physics_schedule='staged_300_700')
HEAT_VARIANTS={
    'BASELINE':dict(grad_clip_norm=1.),'CLIP_2':dict(grad_clip_norm=2.),
    'CLIP_5':dict(grad_clip_norm=5.),
    'EARLY_TIME':dict(grad_clip_norm=1.,query_time_policy='early_stratified_v1'),
}

def validate_variant(architecture,label,controls):
    require(set(controls)<={'physics_weights','physics_schedule','query_time_policy','grad_clip_norm','gradient_diagnostics_every'},'unknown study variant control')
    require(architecture=='C_GEO' or 'physics_weights' not in controls,'physics controls require C_GEO')
    require(label in C_VARIANTS or label in {'STAGED_001',*HEAT_VARIANTS,'EARLY_TIME_BEST_CLIP'},'unknown study label')

def physics_weights(config,successful=0):
    weights=config.physics_weights
    if weights is None:weights=(0.,0.,0.) if config.objective_variant=='FIELD_ONLY' or config.architecture!='C_GEO' else (.1,.1,.1)
    factor=1. if config.physics_schedule=='constant' else max(0.,min(1.,(successful-300)/400))
    return tuple(float(w)*factor for w in weights)

def configured_adapter(config,successful=0):
    from .query_loss import QueryAdapter
    weights=physics_weights(config,successful)
    weak=WeakConfig(config.spatial_order,config.time_order,config.weak_tests,config.physics_chunk,
                    weak_weight=weights[0],initial_weight=weights[1],boundary_weight=weights[2])
    return QueryAdapter(config.architecture,weak,chunk=config.query_chunk,checkpointing=config.query_checkpointing,
        field_only=config.architecture!='C_GEO' or not any(weights),amp=config.resolved_amp_mode)

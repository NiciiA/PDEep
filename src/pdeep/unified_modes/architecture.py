"""Versioned G1 field-conditioned capacities; legacy backbone presets stay intact."""
from dataclasses import asdict,replace
from ..core import require
from ..geometric.models import ModelConfig

LARGE_MINIMUM=20_000_000

def model_config(architecture,capacity):
    require(capacity in ('verification','medium','large'),'unsupported G1 capacity')
    old=ModelConfig.preset(architecture,capacity)
    # Keep320 geometric memory tokens: increasing attention coverage quadratically
    # is unnecessary. Width384 gives26.7M/27.9M connected A/C parameters.
    return replace(old,width=384,heads=8) if capacity=='large' else old

def inventory(model):
    c=model.config
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if c.capacity=='large':require(trainable>=LARGE_MINIMUM,'large G1 model has fewer than20M trainable parameters')
    return {'architecture':c.architecture,'capacity':c.capacity,'model_config':asdict(c),
            'total_parameters':total,'trainable_parameters':trainable,
            'components':{k:sum(p.numel() for p in m.parameters()) for k,m in model.named_children()},
            'MLP_expansion':{'attention':4,'A_decoder':2,'C_FiLM_decoder':2},
            'memory_tokens':{'coarse_cap':c.coarse,'fine_cap':c.fine,'scope':'actual anchors limited by geometry mesh; no trainable anchor table'},
            'float32_weights_MB':4*total/1e6,'float32_weights_grad_Adam_MB':16*total/1e6,
            'optimizer_budget_scope':'weights + gradients + two Adam moments; excludes activations/foreach temporaries/runtime',
            'instances':'independent trainable instances; one shared network per model across all24 tasks and3 modes'}

def verify_definition(record,model):
    """Full definition check, even when an old preset also used the name large."""
    require(record['model_config']==asdict(model.config),'saved architecture definition differs; no partial capacity transfer')
    measured=inventory(model)
    if model.config.capacity=='large':
        require(record.get('parameter_counts')=={k:measured[k] for k in ('total_parameters','trainable_parameters')},'large checkpoint parameter inventory missing or inconsistent')
    return measured

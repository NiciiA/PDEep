"""Well-formed synthetic resume payloads, without optimizer steps."""
import torch
from pdeep.artifact_rng import capture_rng

def states(model,config,successful,observations):
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    return {'optimizer':optimizer.state_dict(),'scheduler':{'kind':config.scheduler,'completed':successful,'next_lr':config.learning_rate},
            'rng':capture_rng(),'sampler':{'observations':observations,'config':config.identity,'cursor':successful},
            'training_state':{'successful':successful,'sampler_cursor':successful}}

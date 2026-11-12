"""One real16k medium-model forward/backward preflight, ZERO optimizer updates."""
import argparse
from pathlib import Path
import torch
from pdeep.artifact_io import atomic_json
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.learning_diagnostics.workflows import prepare_diagnostic,performance_preflight
from pdeep.learning_diagnostics.study_controls import HEAT_VARIANTS


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cpu');parser.add_argument('--output',default='reports/optimization_diagnostics/preflight-cpu');args=parser.parse_args()
    def forbidden(*a,**kw):raise AssertionError('No training/qualification allowed')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):cls.step=forbidden
    import pdeep.smooth.qualification as q
    q.ensure_compatible_certificate=forbidden
    certificate=resolve_certification(requested_capabilities={'certification_id':None}).require()
    root=Path(args.output)
    plan=prepare_diagnostic(model='A_GEO',experiment='heat_single',capacity='medium',updates=1000,device=args.device,certificate=certificate,root=root,
        query_mode='dense_resampled',query_count=16384,query_chunk=4096,query_checkpointing=False,query_revision='continuous-queries-v3',
        query_time_policy='stratified',evaluation_times=(0.,.05,.1,.25,.5,.75,1.),amp_mode='auto',advanced={'grid_size':4},variant_settings={'BASELINE':HEAT_VARIANTS['BASELINE']})
    result={'parameters':plan.models[0].parameter_count,'device':args.device,'actual_CUDA_available':torch.cuda.is_available(),
        'optimizer_updates':0,'qualifications':0,'live_Google_Drive_tested':False,'profiles':performance_preflight(plan)}
    atomic_json(root/'verification.json',result)
    print('Actual',args.device,'medium16k preflight verified; zero optimizer updates.')

if __name__=='__main__':main()

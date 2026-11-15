"""Extend the existing one-display scalar monitor with mode/exposure denominators."""
from collections import defaultdict,Counter
import statistics
from ..unified_bayes.dashboard import Dashboard as Base

class Dashboard(Base):
    def __init__(self,*a,**k):super().__init__(*a,**k);self.exposures=Counter();self._exposure_update=-1
    def objectives(self,value):
        super().objectives(value);rows=[r for micro in value.get('microbatches',[]) for r in micro.get('per_case',[])]
        if value.get('update',0)>self._exposure_update:
            for r in rows:
                for axis,key in (('PDE',r['pde']),('geometry',r['geometry']),('mode',r['mode']),('observations',r['observations']),('known_fraction',r['known_fraction'])):self.exposures[f'{axis}/{key}']+=1
                self.exposures[f"task-mode/{r['pde']}/{r['geometry']}/{r['mode']}"]+=1
            self._exposure_update=value.get('update',0)
        groups=defaultdict(list)
        for update in self.recent_pde:
            for r in update:
                if 'mode' in r:groups[r['mode']].append(r['unweighted']['field']);groups[r['mode']+'/'+r['pde']].append(r['unweighted']['field'])
        self.latest.update(recent_mode_reconstruction={k:{'mean':statistics.mean(v),'physical_cases':len(v),'weighting':'case, then active quantity and valid queries'} for k,v in groups.items()},cumulative_exposure=dict(self.exposures))
        if value.get('update',0)%10==0:self.render()
    def state_dict(self):return {**super().state_dict(),'mode_exposures':dict(self.exposures),'exposure_through_update':self._exposure_update}
    def restore_state(self,value,**kwargs):
        super().restore_state(value,**kwargs);self.exposures=Counter((value or {}).get('mode_exposures',{}));self._exposure_update=(value or {}).get('exposure_through_update',-1)
    def phase(self,stage,**details):
        if stage=='Training' and 'stop_after' in details:
            details['execution_segment_stop']=details.pop('stop_after');details['requested_session_stop']=self.latest.get('session_stop')
        if stage in ('Model construction','Constructing model'):
            self.exposures.clear();self._exposure_update=-1
            self.latest.update(cumulative_exposure={},recent_mode_reconstruction={})
        return super().phase(stage,**details)

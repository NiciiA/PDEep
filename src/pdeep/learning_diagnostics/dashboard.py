"""One throttled display handle; structured reports only, no stream capture."""
from collections import deque
import html, json, time, warnings
from ..train_monitor import monitor_snapshot

class Dashboard:
    def __init__(self, *, notebook=True, interval=.75, renderer=None, progress_callback=None):
        self.notebook=notebook; self.interval=interval; self.renderer=renderer
        self.progress_callback=progress_callback; self.handle=None; self.last_render=-float('inf')
        self.latest={}; self.recent=deque(maxlen=6); self.errors=[]; self.render_count=0
        self._step=None; self._validation=None; self._run=None

    def update(self, config, state, step=None, validation=None):
        if self._run != state.run_id: self._step=None; self._validation=None; self._run=state.run_id; self.diagnostic_metrics={}
        if step is not None: self._step=step
        new_validation=validation is not None and validation is not self._validation
        if validation is not None: self._validation=validation
        v=monitor_snapshot(config,state,self._step,self._validation)
        v.update(experiment=getattr(config,'experiment',config.preset), geometry=config.training_families,
                 progress_percent=100*state.successful/config.updates,
                 diagnostic_metrics=getattr(self,'diagnostic_metrics',{}))
        self.latest=v
        event={'run':state.run_id,'update':state.successful,'phase':state.phase,'checkpoint':state.last_checkpoint,'Drive':v['Drive_sync']}
        if not self.recent or event != self.recent[-1]: self.recent.append(event)
        try:
            if self.progress_callback: self.progress_callback(v)
            now=time.monotonic()
            if now-self.last_render>=self.interval or new_validation or state.phase in {'failed','complete'}:
                self.render(v); self.last_render=now
        except Exception as error:
            # Rendering is not part of the optimizer transaction. Never hide its exception.
            self.errors.append(f'{type(error).__name__}: {error}')
            if len(self.errors)==1: warnings.warn('Dashboard rendering failed; training JSONL/tracebacks remain authoritative: '+self.errors[-1], RuntimeWarning)
        return v

    def render(self,v):
        d=v.get('diagnostic_metrics',{})
        fields={k:v.get(k) for k in ('architecture','experiment','geometry','PDE_mixture','run_id','successful','requested','progress_percent','phase','elapsed_seconds','ETA_seconds','current_objective','rolling_objective','learning_rate','gradient_before_clip','gradient_after_clip','latest_validation','checkpoint','Drive_sync','failed','skipped','last_error')}
        fields.update(raw_losses=v.get('unweighted'),weighted_losses=v.get('weighted'),per_quantity=d.get('field'),sensor=d.get('sensors'))
        text=json.dumps(fields,indent=2,default=str)+ '\nRecent events: '+json.dumps(list(self.recent),default=str)
        self.render_count+=1
        if self.renderer: self.handle=self.renderer(text,self.handle); return
        if self.notebook:
            from IPython.display import display,HTML
            out=HTML('<pre style="white-space:pre-wrap;max-height:480px;overflow:auto">'+html.escape(text)+'</pre>')
            if self.handle is None: self.handle=display(out,display_id=True)
            else: self.handle.update(out)
        # Non-notebook callers inspect .latest; full engine JSONL is already on disk.

"""One in-place dashboard + cached-state heartbeat; no model/RNG access."""
from contextlib import contextmanager
import html,json,threading,time
from ..learning_diagnostics.performance import memory


class Dashboard:
    def __init__(self,config=None,*,notebook=True,renderer=None,heartbeat_seconds=5):
        self.config=config;self.notebook=notebook;self.renderer=renderer;self.interval=heartbeat_seconds
        self.handle=None;self.latest={'title':'G1 — UNIFIED MULTI-PDE TRAINING','stage':'Setup','successful':0,'whole_study_ETA_seconds':None}
        self.start=time.monotonic();self.last_render=0.;self.last_success=-1;self.failed_stage=None;self._stop=threading.Event()
        if config:self.latest.update(dataset_target=config.totals,tasks=len(config.tasks),PDEs=len(config.pdes),geometries=len(config.families))
    def render(self):
        self.latest['elapsed_seconds']=time.monotonic()-self.start;text=json.dumps(self.latest,indent=2,default=str)
        try:
            if self.renderer:self.handle=self.renderer(text,self.handle)
            elif self.notebook:
                from IPython.display import display,HTML
                value=HTML('<pre style="white-space:pre-wrap;max-height:620px;overflow:auto">'+html.escape(text)+'</pre>')
                if self.handle is None:self.handle=display(value,display_id=True)
                else:self.handle.update(value)
        except Exception:print(text,flush=True)
        self.last_render=time.monotonic()
    def phase(self,stage,**details):
        self.latest.update(stage=stage,details=details)
        if 'task' in details and 'prepared' in details:self.latest.setdefault('task_progress',{})[details['task']]=details.copy()
        self.render()
    def update(self,config,state,step=None,validation=None):
        previous=self.latest.get('stage');self.latest.update(current_model=config.model_key,stage=state.phase,successful=state.successful,
            requested=config.updates,cases_processed=state.cases_processed,learning_rate=state.scheduler_state.get('next_lr'),
            checkpoint=state.last_checkpoint,persistence=state.persistence,last_error=state.last_error)
        if step:
            durations=[r.get('details',{}).get('measured_update_wall_seconds',sum(r['timings'].values())) for r in state.recent_steps if r['status']=='successful'];rate=len(durations)/sum(durations) if durations and sum(durations)>0 else None
            self.latest.update(losses={k:step.unweighted.get(k,'N/A') for k in ('field','auxiliary','weak','initial','boundary','kinematic')},
                weighted_losses=step.weighted,total_weighted_loss=step.total,gradient_norm=step.gradient_before_clip,
                clipping_fraction=sum(r['gradient_before_clip']>config.grad_clip_norm for r in state.recent_steps)/max(1,len(state.recent_steps)),
                updates_per_second=rate,queries_per_second=rate*4*config.query_count if rate else None,
                current_model_ETA_seconds=max(0,self.latest.get('session_stop',config.updates)-state.successful)/rate if rate else None,
                full_schedule_ETA_seconds=(config.updates-state.successful)/rate if rate else None,
                task_mix=state.pde_counts,device_memory=memory(config.device))
        if self.latest.get('model_index')==f"{self.latest.get('model_total')}/{self.latest.get('model_total')}":self.latest['whole_study_ETA_seconds']=self.latest.get('current_model_ETA_seconds')
        if validation:self.latest['validation']=validation.metrics
        if state.phase in ('failed','complete') or state.successful>=self.last_success+10 or previous!=state.phase and state.phase not in ('forward','backward','optimizer','ready','data') or time.monotonic()-self.last_render>=self.interval:
            self.render();self.last_success=state.successful
    @contextmanager
    def live(self):
        self.render();self._stop.clear()
        def heartbeat():
            while not self._stop.wait(self.interval):self.render()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:yield self
        except BaseException as error:
            self.failed_stage=self.latest.get('stage');self.latest.update(failed_stage=self.failed_stage,error=f'{type(error).__name__}: {error}');self.render();raise
        finally:self._stop.set();thread.join(timeout=1)

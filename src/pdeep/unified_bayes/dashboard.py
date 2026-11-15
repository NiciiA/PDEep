"""Observability only: Python scalar records, cached hardware, UI heartbeat."""
from collections import deque,defaultdict
from contextlib import contextmanager
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import html,json,statistics,threading,time
from ..learning_diagnostics.performance import memory,utilization

ALIASES={'Dataset inspection':'Preparing dataset','Dataset cache loading':'Loading local cache','Reference preparation':'Preparing references',
         'Model construction':'Constructing model','Numerical preflight':'Memory preflight','Checkpoint saving':'Saving checkpoint',
         'validation':'Validating','saving':'Saving checkpoint','complete':'Completed'}
LOSSES=('field','auxiliary','kinematic','weak','initial','boundary')

class Dashboard:
    def __init__(self,config=None,*,notebook=True,renderer=None,heartbeat_seconds=5,warmup=10,timezone='Europe/Stockholm'):
        self.config=config;self.notebook=notebook;self.renderer=renderer;self.interval=heartbeat_seconds;self.warmup=warmup;self.timezone=timezone
        self.handle=None;self.latest={'title':'PDEep — G1 UNIFIED TRAINING','scope':'UNIFIED MULTI-PDE','stage':'Setup','successful':0,'whole_study_ETA_seconds':None,'ETA_status':'collecting measurements'}
        self.start=time.monotonic();self.last_render=0.;self.last_success=-1;self.failed_stage=None;self._stop=threading.Event();self.render_count=0
        self.run=None;self.seen=set();self.unique_complete=True;self.last_recorded=-1;self.recent_pde=deque(maxlen=50);self.save_durations=deque(maxlen=50);self.hardware={};self.hardware_at=-float('inf');self.completed=set()
        if config:self.latest.update(dataset_target=config.totals,tasks=len(config.tasks),PDEs=len(config.pdes),geometries=len(config.families),dataset=config.preset)
    def _hardware(self,device):
        if self.hardware.get('device')==str(device) and time.monotonic()-self.hardware_at<5:return
        self.hardware=memory(device);self.hardware['utilization']=utilization(device);self.hardware_at=time.monotonic()
    def render(self):
        self.latest['cell_elapsed_seconds']=time.monotonic()-self.start
        text=json.dumps(self.latest,indent=2,default=str);self.render_count+=1
        try:
            if self.renderer:self.handle=self.renderer(text,self.handle)
            elif self.notebook:
                from IPython.display import display,HTML
                view=HTML('<pre style="white-space:pre-wrap;max-height:670px;overflow:auto">'+html.escape(text)+'</pre>')
                if self.handle is None:self.handle=display(view,display_id=True)
                else:self.handle.update(view)
        except Exception:print(text,flush=True)
        self.last_render=time.monotonic()
    def phase(self,stage,**details):
        self.latest.update(stage=stage,display_stage=ALIASES.get(stage,stage),details=details)
        if ALIASES.get(stage,stage)=='Completed' and 'models' in details:self.latest.update(completed_models=details['models'],whole_study_ETA_seconds=0.)
        if ALIASES.get(stage,stage)=='Constructing model':
            for key in ('last_update_seconds','median_update_seconds','estimated_completion','full_schedule_ETA_seconds','recent_PDE_weighted_objectives','recent_PDE_cases'):self.latest.pop(key,None)
            self.latest['ETA_status']='collecting measurements'
        if 'task' in details and 'prepared' in details:self.latest.setdefault('task_progress',{})[details['task']]=dict(details)
        self.render()
    def objectives(self,value):
        rows=[r for micro in value.get('microbatches',[]) for r in micro.get('per_case',[])]
        if rows:self.recent_pde.append(rows)
        groups=defaultdict(list)
        for update in self.recent_pde:
            for r in update:groups[r['pde']].append(sum(r['weighted'].values()))
        self.latest['recent_PDE_weighted_objectives']={p:statistics.mean(v) for p,v in groups.items()}
        self.latest['recent_PDE_cases']={p:len(v) for p,v in groups.items()}
        if value.get('update',0)%10==0:self.render()
    def saved(self,seconds,status):
        self.save_durations.append(float(seconds));self.latest.update(last_checkpoint_save_sync_seconds=float(seconds),last_Drive_sync_seconds=status.get('seconds'),last_Drive_status=status.get('Drive_status'))
    def state_dict(self):
        return {'schema':'G1-monitor-scalars-v1','run':self.run,'seen_case_ids':sorted(self.seen),'unique_complete':self.unique_complete,
                'last_recorded':self.last_recorded,'measured_save_durations':list(self.save_durations),'recent_PDE_objectives':list(self.recent_pde)}
    def restore_state(self,value,*,run_id=None,history=None,successful=0):
        self.run=run_id or (value or {}).get('run');self.recent_pde.clear();self.save_durations.clear()
        if value:
            self.seen=set(value.get('seen_case_ids',[]));self.unique_complete=value.get('unique_complete',False)
            self.recent_pde.extend(value.get('recent_PDE_objectives',[]));self.save_durations.extend(value.get('measured_save_durations',[]));self.last_recorded=value.get('last_recorded',successful)
        else:
            records=(history or {}).get('steps.jsonl',[]);self.seen={c for r in records for c in r.get('case_ids',[])}
            self.unique_complete=bool(records) or successful==0;self.last_recorded=successful
            for value in (history or {}).get('case_losses.jsonl',[])[-50:]:
                rows=[r for micro in value.get('microbatches',[]) for r in micro.get('per_case',[])]
                if rows:self.recent_pde.append(rows)
        self.latest['unique_physical_cases']=len(self.seen) if self.unique_complete else 'N/A: prior exposure history unavailable'
    def update(self,config,state,step=None,validation=None):
        if self.run!=state.run_id:
            self.run=state.run_id;self.seen=set();self.unique_complete=state.successful==0 or bool(step and step.successful==1);self.last_recorded=-1;self.recent_pde.clear();self.save_durations.clear();self.last_success=-1
            for key in ('validation','recent_PDE_weighted_objectives','current_model_ETA_seconds','estimated_completion') :self.latest.pop(key,None)
        stage=ALIASES.get(state.phase,'Training' if state.phase in ('data','forward','backward','optimizer','ready') else state.phase)
        previous=self.latest.get('stage');self._hardware(config.device)
        self.latest.update(current_model=config.model_key,stage=stage,successful=state.successful,requested=config.updates,
            pilot={'successful':min(3000,state.successful),'target':min(3000,config.updates)},cases_processed=state.cases_processed,
            case_exposures=state.cases_processed,approximate_dataset_passes=state.cases_processed/max(1,self.config.totals['train']) if self.config else None,
            learning_rate=state.scheduler_state.get('next_lr'),checkpoint=state.last_checkpoint,last_verified_checkpoint=state.last_checkpoint,
            best_validation_checkpoint=state.best_checkpoint,persistence=state.persistence,elapsed_seconds=state.elapsed_seconds,last_error=state.last_error,
            AMP='FP32 (auto resolves off for this verified training engine)',AMP_loss_scale='N/A (no scaler)',device=config.device,device_memory=self.hardware,
            checkpoint_save_seconds_total=state.saving_seconds,validation_seconds_total=state.validation_seconds)
        if step and step.status=='successful' and step.successful>self.last_recorded:
            self.seen.update(step.case_ids);self.last_recorded=step.successful
        self.latest['unique_physical_cases']=len(self.seen) if self.unique_complete else 'N/A: prior exposure history unavailable'
        if step:
            recent=[r for r in state.recent_steps if r['status']=='successful'];durations=[r.get('details',{}).get('measured_update_wall_seconds',sum(r['timings'].values())) for r in recent]
            median=statistics.median(durations) if durations else None
            self.latest.update(losses={k:step.unweighted.get(k,'N/A (inactive)') for k in LOSSES},weighted_losses={k:step.weighted.get(k,'N/A (inactive)') for k in LOSSES},
                total_weighted_loss=step.total,gradient_norm=step.gradient_before_clip,gradient_after_clip=step.gradient_after_clip,
                clipping_fraction=sum(r['gradient_before_clip']>config.grad_clip_norm for r in recent)/len(recent) if recent else None,
                last_update_seconds=durations[-1] if durations else None,median_update_seconds=median,
                updates_per_second=1/median if median else None,queries_per_second=config.effective_batch*config.query_count/median if median else None,task_mix=state.pde_counts,
                kinematic_scope='wave cases only' if config.model_key=='A_GEO/KIN_001' else 'N/A')
            target=self.latest.get('session_stop',config.updates)
            if len(durations)>=self.warmup:
                val_average=state.validation_seconds/len(state.validation_history) if state.validation_history else None
                save_average=statistics.mean(self.save_durations) if self.save_durations else None
                def estimate(goal):
                    remaining=max(0,goal-state.successful)*median
                    if val_average is not None:remaining+=max(0,goal//config.validation_every-state.successful//config.validation_every)*val_average
                    if save_average is not None:remaining+=max(0,goal//config.checkpoint_every-state.successful//config.checkpoint_every)*save_average
                    return remaining
                eta=estimate(target);self.latest.update(current_model_ETA_seconds=eta,full_schedule_ETA_seconds=estimate(config.updates),ETA_status='measured',
                    ETA_basis={'statistic':'median of recent successful updates','samples':len(durations),'window':config.rolling_window,
                               'mean_validation_seconds':val_average,'mean_save_sync_seconds':save_average,'unmeasured_overhead':'excluded and N/A, never fabricated'},
                    estimated_completion=(datetime.now(ZoneInfo(self.timezone))+timedelta(seconds=eta)).isoformat())
            else:self.latest.update(current_model_ETA_seconds=None,full_schedule_ETA_seconds=None,ETA_status='collecting measurements',estimated_completion=None)
        if validation:self.latest['validation']=validation.metrics
        self.latest['whole_study_ETA_seconds']=self.latest.get('current_model_ETA_seconds') if self.latest.get('model_index')==f"{self.latest.get('model_total')}/{self.latest.get('model_total')}" else None
        self.latest['whole_study_ETA_status']='measured active final model' if self.latest.get('whole_study_ETA_seconds') is not None else 'N/A: remaining architectures have unmeasured costs'
        if state.phase=='complete':self.completed.add(state.run_id)
        if state.phase in ('failed','complete') or state.successful>=self.last_success+10 or previous!=stage and stage not in ('Training',) or time.monotonic()-self.last_render>=self.interval:
            self.render();self.last_success=state.successful
    @contextmanager
    def live(self):
        self.render();self._stop.clear()
        def heartbeat():
            while not self._stop.wait(self.interval):self.render()
        thread=threading.Thread(target=heartbeat,daemon=True,name='G1-display-heartbeat');thread.start()
        try:yield self
        except BaseException as error:
            self.failed_stage=self.latest.get('stage');self.latest.update(failed_stage=self.failed_stage,error=f'{type(error).__name__}: {error}');self.render();raise
        finally:self._stop.set();thread.join(timeout=1)

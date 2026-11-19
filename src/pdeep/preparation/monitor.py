"""Explicit operation clocks for preparation/preflight, no training state."""
from contextlib import contextmanager
import time
from ..unified_modes.dashboard import Dashboard as Base

class Dashboard(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._stage_start=self.start;self._active_operation=None;self._last_operation_seconds=0.
        self.latest.update(cell_elapsed_scope='legacy alias of monitor lifetime, not stage work; preflight owns a fresh monitor',completed_operation_timings=[])
    def phase(self,stage,**details):
        if stage!=self.latest.get('stage'):
            self._stage_start=time.monotonic();self.latest.pop('reported_stage_seconds',None)
        if 'seconds' in details:self.latest['reported_stage_seconds']=details['seconds']
        return super().phase(stage,**details)
    def render(self):
        now=time.monotonic();self.latest['monitor_elapsed_seconds']=now-self.start;self.latest['stage_elapsed_seconds']=now-self._stage_start
        self.latest['operation_elapsed_seconds']=now-self._active_operation[1] if self._active_operation else self._last_operation_seconds
        self.latest['active_operation']=self._active_operation[0] if self._active_operation else None
        return super().render()
    @contextmanager
    def operation(self,name,**details):
        previous=self._active_operation;start=time.monotonic();self._active_operation=(name,start);self.phase(name,**details)
        status='completed'
        try:yield
        except BaseException:status='failed';raise
        finally:
            seconds=time.monotonic()-start;self._last_operation_seconds=seconds;self._active_operation=previous
            history=self.latest['completed_operation_timings'];history.append({'operation':name,'seconds':seconds,'status':status,**details});del history[:-128]
            self.render()

"""Event-only notebook launch. Executing a cell/Run All never calls its callback.

No persisted boolean/string is a launch authorization. A fresh button click is
required. Verification code must never simulate clicks; CLI training is separate.
"""
from dataclasses import dataclass


@dataclass
class LaunchButton:
    callback: object
    description: str='Start reviewed training'
    enabled: bool=True
    widget: object=None
    def render(self):
        import ipywidgets as widgets
        from IPython.display import display
        button=widgets.Button(description=self.description,disabled=not self.enabled,button_style='warning')
        output=widgets.Output()
        def clicked(_):
            if not self.enabled:return
            button.disabled=True
            try:
                with output:self.callback()
            finally:button.disabled=not self.enabled
        button.on_click(clicked);self.widget=button;display(button,output);return self


def offer_launch(callback,*,description='Start reviewed training',enabled=True):
    """Render only; neither imports trainers nor invokes callbacks."""
    return LaunchButton(callback,description,enabled).render()

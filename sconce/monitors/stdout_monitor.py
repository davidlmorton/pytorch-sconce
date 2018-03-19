from sconce.monitors.base import Monitor
from sconce.utils import Progbar
from torch.autograd import Variable


class StdoutMonitor(Monitor):
    def __init__(self, metric_names, progbar_kwargs={},
            name='stdout_monitor'):
        super().__init__(name=name)
        self._metric_names = metric_names
        self._progbar_kwargs = progbar_kwargs
        self._progress_bar = None

    def start_session(self, num_steps):
        self._progress_bar = Progbar(num_steps, **self._progbar_kwargs)

    def step(self, data):
        if self._progress_bar is None:
            raise RuntimeError("You must call 'start_session' before "
                    "calling 'step'")

        values = []
        for key, name in self._metric_names.items():
            if key in data.keys():
                value = data[key]
                if isinstance(value, Variable):
                    value = value.data[0]
                values.append((name, value))
        self._progress_bar.add(1, values)

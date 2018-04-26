from sconce.monitors.base import Monitor
from sconce.utils import Progbar


class StdoutMonitor(Monitor):
    def __init__(self, metric_names, progbar_kwargs={},
            name='stdout_monitor'):
        super().__init__(name=name)
        self._metric_names = metric_names
        self._progbar_kwargs = progbar_kwargs
        self._progress_bar = None

    def start_session(self, num_steps, **kwargs):
        self._progress_bar = Progbar(num_steps, **self._progbar_kwargs)

    def write(self, data, step, **kwargs):
        if step != int(step):
            # don't update for partial steps
            return

        if self._progress_bar is None:
            raise RuntimeError("You must call 'start_session' before "
                    "calling 'write'")

        values = []
        for key, name in self._metric_names.items():
            if key in data.keys():
                value = data[key]
                value = self._to_scalar(value)
                values.append((name, value))
        self._progress_bar.add(1, values)

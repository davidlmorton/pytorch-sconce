from abc import ABC, abstractmethod


class Monitor(ABC):
    def __init__(self, name):
        self.name = name

    def start_session(self, num_steps, **kwargs):
        pass

    def end_session(self):
        pass

    @abstractmethod
    def step(self, data):
        pass

    def __add__(self, other):
        return CompositeMonitor(monitors=[self, other])


class CompositeMonitor(Monitor):
    def __init__(self, monitors, name='composite_monitor'):
        super().__init__(name=name)

        self.monitors = []

        for monitor in monitors:
            self.add_monitor(monitor)

    def add_monitor(self, other):
        if isinstance(other, CompositeMonitor):
            for nested_monitor in other.monitors:
                self._add_monitor(nested_monitor)
        else:
            self._add_monitor(other)

    def _add_monitor(self, other):
        if hasattr(self, other.name):
            raise RuntimeError("A monitor with the name "
                    f"'{other.name}' cannot be added to this "
                    "CompositeMonitor.  Have you already added it?")
        else:
            self.monitors.append(other)
            setattr(self, other.name, other)

    def start_session(self, num_steps, **kwargs):
        for monitor in self.monitors:
            monitor.start_session(num_steps, **kwargs)

    def end_session(self):
        for monitor in self.monitors:
            monitor.end_session()

    def step(self, data):
        for monitor in self.monitors:
            monitor.step(data)

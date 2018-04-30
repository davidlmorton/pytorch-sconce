from abc import ABC, abstractmethod
from torch import Tensor


class Monitor(ABC):
    """
    Base class for monitors in sconce.  A monitor is an object that a :py:class:`~sconce.trainer.Trainer` uses to
    record metrics during training or other tasks.  This base class defines the interface that trainers use.  Monitors
    can be composed together (using addition operator) to produce a :py:class:`~sconce.monitors.base.CompositeMonitor`
    object.

    Arguments:
        name (str): used to gain access to a monitor when it has been composed together into a
            :py:class:`~sconce.monitors.base.CompositeMonitor`.
    """
    def __init__(self, name):
        self.name = name

    def _to_scalar(self, value):
        if isinstance(value, Tensor):
            try:
                return value.item()
            except ValueError:
                return value.data[0].cpu().numpy()
        else:
            return value

    def start_session(self, num_steps, **kwargs):
        """
        Called by a :py:class:`~sconce.trainer.Trainer` when starting a training/evaluation session.

        Arguments:
            num_steps (int): [1, inf) the number of update/evaluation steps to expect.
            **kwargs:  must be accepted to allow for future use cases.
        """
        pass

    def end_session(self, **kwargs):
        """
        Called by a :py:class:`~sconce.trainer.Trainer` when a training/evaluation session has ended.

        Arguments:
            **kwargs: must be accepted to allow for future use cases.
        """
        pass

    @abstractmethod
    def write(self, data, step, **kwargs):
        """
        Called by a :py:class:`~sconce.trainer.Trainer` during a
        training/evaluation session just after the training/evaluation step.

        Arguments:
            data (dict): the output of the training/evaluation step.  The keys
                may include, but are not limited to: {'training_loss', 'test_loss',
                'learning_rate'}.
            step (float): (0.0, inf) the step that was just completed.
                Fractional steps are possible (see batch_multiplier option on
                :py:meth:`sconce.trainer.Trainer.train`).
            **kwargs: must be accepted to allow for future use cases.
        """
        pass

    def __add__(self, other):
        return CompositeMonitor(monitors=[self, other])


class CompositeMonitor(Monitor):
    """
    A monitor composed of two or more monitors.  Using this allows you to pass a single monitor object to a trainer
    method and have it use all of the composed monitors.  Composed monitors can be accessed using their name like so:

        >>> from sconce import monitors
        >>> metric_names = {'training_loss': 'loss', 'test_loss': 'val_loss'}
        >>> stdout_monitor = monitors.StdoutMonitor(metric_names=metric_names)
        >>> dataframe_monitor = monitors.DataframeMonitor()
        >>> monitor = dataframe_monitor + stdout_monitor
        >>> monitor.dataframe_monitor
        <sconce.monitors.dataframe_monitor.DataframeMonitor at 0x7fb1fbd498d0>
        >>> dataframe_monitor is monitor.dataframe_monitor
        True

    Arguments:
        monitors (iterable of :py:class:`~sconce.monitors.base.Monitor`): the monitors you want to compose together.
    """
    def __init__(self, monitors):
        super().__init__(name=None)

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

    def end_session(self, **kwargs):
        for monitor in self.monitors:
            monitor.end_session(**kwargs)

    def write(self, data, step, **kwargs):
        for monitor in self.monitors:
            monitor.write(data, step, **kwargs)

from pprint import pformat as pf
from abc import ABC, abstractmethod


class Schedule(ABC):
    """
    The base class for all schedules in Sconce.  It is only an interface, describing what must be implemented
    if you want to define a schedule.
    """
    def set_num_steps(self, value):
        """
        This is the canonical method to set num_steps, taking care of setting any other state needed
            by the schedule as well.
        """
        self.num_steps = value

    def get_value(self, step, current_state):
        """
        Returns the value one should set, based on this schedule.

        Arguments:
            step (float): (0.0, inf) the training step that is about to be completed.
                Fractional steps are possible (see batch_multiplier option on
                :py:meth:`sconce.trainer.Trainer.train`).
            current_state (dict): a dictionary describing the current training state.
        """
        if not hasattr(self, 'num_steps'):
            raise RuntimeError("You should not call Schedule.get_value() before setting num_steps.")
        elif step > self.num_steps:
            raise RuntimeError(f"Argument step={step}, should not exceed num_steps={self.num_steps}")
        else:
            return self._get_value(step, current_state)

    @abstractmethod
    def __repr__(self):
        """
        Output a string that could be eval'd to reproduce this object.
        For Example: ``Triangle(initial_value=1, peak_value=10)``
        """


class ScheduledMixin:
    """
    This mixin defines the interface for scheduled objects in Sconce.
    """
    def __init__(self):
        self.schedules = {}

    def set_schedule(self, name, schedule):
        set_method_name = f'set_{name}'
        if not hasattr(self, set_method_name):
            raise RuntimeError(f'Cannot set schedule for attribute named ({name}), because no set'
                f'method ({set_method_name}) is defined on this class ({self.__class__.__name__})')

        # if they just passed in a value, make it a constant schedule instead
        if not isinstance(schedule, Schedule):
            schedule = Constant(value=schedule)

        self.schedules[name] = schedule
        return self.schedules

    def remove_schedule(self, name):
        if name in self.schedules:
            del self.schedules[name]

    def start_session(self, num_steps):
        for schedule in self.schedules.values():
            schedule.set_num_steps(num_steps)

    def prepare_for_step(self, step, current_state):
        hyperparameters = {}
        for name, schedule in self.schedules.items():
            set_method_name = f'set_{name}'
            set_method = getattr(self, set_method_name)
            value = schedule.get_value(step=step, current_state=current_state)
            set_method(value)
            hyperparameters[name] = value
        return hyperparameters


class Constant(Schedule):
    """
    A schedule where the value is always the same.

    Arguments:
        value (float): the value of the hyperparameter.
    """
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'{self.__class__.__name__}(value={pf(self.value)})'

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def _get_value(self, step, current_state):
        return self.value

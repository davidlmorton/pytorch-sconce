from abc import ABC, abstractmethod
from sconce.parameter_group import ParameterGroup
from sconce.schedules.base import ScheduledMixin
from torch import nn

import numpy as np


class Model(ABC, nn.Module, ScheduledMixin):
    """
    The base class of all Models in Sconce.  It is only an interface, describing what must be implemented
    if you want to define a model.
    """
    def __init__(self):
        super(ABC, self).__init__()
        super(nn.Module, self).__init__()
        super(ScheduledMixin, self).__init__()

        self._parameter_groups = {}
        self.default_parameter_group_name = '__default__'

    def build_parameter_groups(self):
        """
        This can be overridden to build additional parameter groups.  This can be useful if you're doing
        different layerwise optimization schedules.
        """
        parameters = self.get_trainable_parameters()
        group = ParameterGroup(parameters=parameters, name=self.default_parameter_group_name)
        self.add_parameter_group(group)

    def add_parameter_group(self, group, inactivate_default=True):
        """
        Add a new parameter group to this model.

        Arguments:
            group (:py:class:`~sconce.parameter_group.ParameterGroup`): the parameter group to add.
            inactivate_default (bool, optional): if ``True``, then the default parameter group will have
                ``is_active`` set to ``False``.
        """
        self._parameter_groups[group.name] = group
        if inactivate_default and group.name != self.default_parameter_group_name:
            self.default_parameter_group.is_active = False

    @abstractmethod
    def forward(self, *, inputs, targets, **kwargs):
        """
        It must accept arbitrary keyword arguments.  The base class of trainer will pass
        `inputs` and `targets`, but subclasses may modify that behavior to include other keyword arguments.

        It must return a dictionary. The dictionary is expected to include at least the key `outputs`
        but may include any other keys you like.  The value of the key `outputs` is expected to be
        the :py:class:`torch.Tensor` output of the model, used for calculating the loss.
        """

    @abstractmethod
    def calculate_loss(self, *, inputs, outputs, targets, **kwargs):
        """
        This method must accept arbitrary keyword arguments.  The base class of trainer will pass `inputs`,
        `outputs`, and `targets`, but subclasses may modify that behavior to include other keyword arguments.

        It must return a dictionary.  The dictionary is expected to include at least the key 'loss', but may
        include any otehr keys you like.  The value of the key `loss` is expected to be the :py:class:`torch.Tensor`
        output of the loss function, used to back-propagate the gradients used by the optimizer.
        """

    def calculate_metrics(self, *, inputs, outputs, targets, loss, **kwargs):
        """
        This method must accept arbitrary keyword arguments.  The base class of trainer will pass `inputs`,
        `outputs`, `targets`, and `loss`, but subclasses may modify that behavior to include other keyword arguments.

        It must return a dictionary.  No restrictions are made on the keys or values of this dictionary.
        """
        return {}

    def get_optimizers(self):
        """
        Returns a list of optimizers for the parameters of this model.
        """
        result = []
        for group in self.active_parameter_groups:
            if group.optimizer is not None:
                result.append(group.optimizer)

        if not result:
            raise RuntimeError("No active parameter groups with optimizers found. "
                    "Did you add an optimizer with 'set_optimizer'?")
        return result

    def get_trainable_parameters(self):
        """
        The trainable parameters that the models has.
        """
        return list(filter(lambda p: p.requires_grad, self.parameters()))

    def get_num_trainable_parameters(self):
        """
        The number of trainable parameters that the models has.
        """
        return sum([np.prod(p.size()) for p in self.get_trainable_parameters()])

    def prepare_for_step(self, step, current_state):
        """
        First, it handles any hyperparameter schedules added to the model itself before gathering up
        the results of calling 'prepare_for_step' on all the model's parameter groups and combining the result.
        """
        model_hyperparameters = super().prepare_for_step(step=step, current_state=current_state)
        hyperparameters = {'model': model_hyperparameters}
        for group in self.active_parameter_groups:
            group_hyperparameters = group.prepare_for_step(step=step, current_state=current_state)
            hyperparameters[group.name] = group_hyperparameters
        return hyperparameters

    def set_schedule(self, name, schedule):
        """
        Set the schedule for a hyperparameter on this model.

        Arguments:
            name (string): the name of the hyperparameter you want to schedule.
            schedule (:py:class:~sconce.schedules.base.Schedule): the schedule for that hyperparameter.

        Note:
            Some name values are interpreted specially.  Setting name to 'learning_rate', 'momentum', or 'weight_decay'
            will delegate to setting schedules on all active parameter groups instead of on the model.
        """
        if name in ('learning_rate', 'momentum', 'weight_decay'):
            for group in self.active_parameter_groups:
                group.set_schedule(name=name, schedule=schedule)
        else:
            super().set_schedule(name=name, schedule=schedule)

    @property
    def parameter_groups(self):
        """
        A list of all parameter groups, inactive as well as active.
        """
        if not self._parameter_groups:
            self.build_parameter_groups()
        return self._parameter_groups.values()

    def get_parameter_group(self, name):
        """
        Get a parameter group by name.
        """
        if not self._parameter_groups:
            self.build_parameter_groups()
        return self._parameter_groups[name]

    @property
    def default_parameter_group(self):
        """
        The default parameter group is created automatically and includes all of the
        trainable parameters for the model.
        """
        return self.get_parameter_group(self.default_parameter_group_name)

    @property
    def active_parameter_groups(self):
        """
        A list of all active parameter groups.
        """
        return [g for g in self.parameter_groups if g.is_active]

    def set_optimizer(self, *args, **kwargs):
        """
        Set the optimizer for all of the active parameter groups on this model.
        """
        for group in self.active_parameter_groups:
            group.set_optimizer(*args, **kwargs)

    def start_session(self, num_steps):
        """
        Called by the :py:class:~sconce.trainer.Trainer when a training session starts.

        Arguments:
            num_steps (int): the number of steps the trainer will take during this training session.
        """
        super().start_session(num_steps)
        for group in self.active_parameter_groups:
            group.start_session(num_steps)

    def print_schedule_summary(self):
        """
        Print out a summary of the scheduled hyperparameters on this model and it's parameter groups.
        """
        for name, schedule in self.schedules.items():
            print(f'model.{name}: {schedule}')

        for group in self.parameter_groups:
            for schedule_name, schedule in group.schedules.items():
                if not group.is_active:
                    print('(inactive) ', end='')
                print(f'{group.name}.{schedule_name}: {schedule}')

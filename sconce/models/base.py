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

        parameters = self.get_trainable_parameters()
        self.default_parameter_group_name = name = '__all__'
        self._parameter_groups = {name: ParameterGroup(parameters=parameters, name=name)}

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
        for group in self._parameter_groups.values():
            if group.is_active and group.optimizer is not None:
                result.append(group.optimizer)
        return result

    def get_trainable_parameters(self):
        """
        The trainable parameters that the models has.
        """
        return filter(lambda p: p.requires_grad, self.parameters())

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
        for name, group in self._parameter_groups.items():
            group_hyperparameters = group.prepare_for_step(step=step, current_state=current_state)
            hyperparameters[name] = group_hyperparameters
        return hyperparameters

    def set_schedule(self, name, schedule):
        if name in ('learning_rate', 'momentum', 'weight_decay'):
            self.default_parameter_group.set_schedule(name=name, schedule=schedule)
        else:
            super().set_schedule(name=name, schedule=schedule)

    @property
    def default_parameter_group(self):
        return self.get_parameter_group(self.default_parameter_group_name)

    def get_parameter_group(self, name):
        return self._parameter_groups[name]

    def set_optimizer(self, *args, **kwargs):
        return self.default_parameter_group.set_optimizer(*args, **kwargs)

    def set_learning_rate(self, *args, **kwargs):
        return self.default_parameter_group.set_learning_rate(*args, **kwargs)

    def set_momentum(self, *args, **kwargs):
        return self.default_parameter_group.set_momentum(*args, **kwargs)

    def set_weight_decay(self, *args, **kwargs):
        return self.default_parameter_group.set_weight_decay(*args, **kwargs)

    def start_session(self, num_steps):
        super().start_session(num_steps)
        for group in self._parameter_groups.values():
            if group.is_active:
                group.start_session(num_steps)

    def print_schedule_summary(self):
        print('model')
        print('=====')
        for name, schedule in self.schedules.items():
            print(f'{name}: {schedule}')

        print('parameter groups')
        print('================')
        for group_name, group in self._parameter_groups.items():
            for schedule_name, schedule in group.schedules.items():
                print(f'{group_name}.{schedule_name}: {schedule}')

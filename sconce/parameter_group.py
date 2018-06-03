from sconce.schedules.base import ScheduledMixin


class ParameterGroup(ScheduledMixin):
    """
    A parameter group is the way that sconce models organize nn.Module parameters and their associated optimizers.

    Arguments:
        parameters (iterable of :py:class:`torch.nn.Parameter`): the parameters you want to group together.
        name (string): your name for this group
        is_active (bool, optional): should this group be considered active (used during training)?
    """
    def __init__(self, parameters, name, is_active=True):
        super().__init__()
        self.parameters = parameters
        self.name = name
        self.optimizer = None
        self.is_active = is_active

    def set_optimizer(self, optimizer_class, *args, **kwargs):
        """
        Set an optimizer on this parameter group.  If this parameter group is active (has ``is_active=True``) then this
        optimizer will be used during training.

        Arguments:
            optimizer_class (one of the :py:mod:`torch.optim` classes): the class of optimizer to set.

        Note:
            All other arguments and keyword arguments are delivered to the optimizer_class's constructor.
        """
        self.optimizer = optimizer_class(params=self.parameters, *args, **kwargs)
        return self.optimizer

    def set_learning_rate(self, desired_learning_rate):
        param_groups = self.optimizer.param_groups
        for param_group in param_groups:
            param_group['lr'] = desired_learning_rate
        return desired_learning_rate

    def set_momentum(self, desired_momentum):
        param_groups = self.optimizer.param_groups
        for param_group in param_groups:
            param_group['momentum'] = desired_momentum
        return desired_momentum

    def set_weight_decay(self, desired_weight_decay):
        param_groups = self.optimizer.param_groups
        for param_group in param_groups:
            param_group['weight_decay'] = desired_weight_decay
        return desired_weight_decay

    def freeze(self):
        """
        Set ``requires_grad = False`` for all parameters in this group.
        """
        self.is_active = False
        for parameter in self.parameters:
            parameter.requires_grad = False

    def unfreeze(self):
        """
        Set ``requires_grad = True`` for all parameters in this group.
        """
        self.is_active = True
        for parameter in self.parameters:
            parameter.requires_grad = True

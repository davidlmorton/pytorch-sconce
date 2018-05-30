from sconce.schedules.base import ScheduledMixin


class ParameterGroup(ScheduledMixin):
    def __init__(self, parameters, name, is_active=True):
        super().__init__()
        self.parameters = parameters
        self.name = name
        self.optimizer = None
        self.is_active = is_active

    def set_optimizer(self, optimizer_class, *args, **kwargs):
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
        pass

    def unfreeze(self):
        pass

from abc import ABC, abstractmethod
from collections import OrderedDict


class RateController(ABC):
    """
    The base class of all rate controllers in Sconce.  It is only an interface, describing what must be implemented
    if you want to define a rate controller.
    """

    @abstractmethod
    def start_session(self, num_steps):
        """
        Called by a :py:class:`~sconce.trainer.Trainer` when starting a training session.

        Arguments:
            num_steps (int): [1, inf) the number of update steps to expect.
        """
        pass

    @abstractmethod
    def new_learning_rate(self, step, data):
        """
        Called by a :py:class:`~sconce.trainer.Trainer` during a
        training/evaluation session just before the training step.

        Arguments:
            data (dict): the output of the training/evaluation step.  The keys
                may include, but are not limited to: {'training_loss', 'test_loss',
                'learning_rate'}.
            step (float): (0.0, inf) the step that was just completed.
                Fractional steps are possible (see batch_multiplier option on
                :py:meth:`sconce.trainer.Trainer.train`).

        Returns:
            new_learning_rate (float, :py:class:`collections.OrderedDict`): The new learning rate that should be used
            for the next training step.  If this is a :py:class:`~sconce.rate_controllers.base.CompositeRateController`
            then an OrderedDict is returned where the keys are like, {'group 0', 'group 1', ect}, and the values are the
            new learning rate (float) for that parameter group.
        """
        pass


class CompositeRateController(RateController):
    """
    A rate controller composed of two or more rate controllers.  Using this allows you to pass a single rate controller
    to a trainer, and control the learning rate of multiple parameter groups.  The order that the controllers are added
    is important, and aligns to the order of the :py:class:`~torch.optim.optimizer.Optimizer`'s parameter_groups.

    Arguments:
        rate_controllers (iterable of :py:class:`~sconce.rate_controllers.base.RateController`): the
            rate_controllers you want to compose together.

    New in 0.9.0
    """
    def __init__(self, rate_controllers):
        super().__init__()

        self.rate_controllers = []

        for rc in rate_controllers:
            self.add_rate_controller(rc)

    def add_rate_controller(self, other):
        if isinstance(other, CompositeRateController):
            for nested_rate_controller in other.rate_controllers:
                self._add_rate_controller(nested_rate_controller)
        else:
            self._add_rate_controller(other)

    def _add_rate_controller(self, other):
        self.rate_controllers.append(other)

    def start_session(self, num_steps):
        for rc in self.rate_controllers:
            rc.start_session(num_steps)

    def new_learning_rate(self, step, data):
        result = OrderedDict()
        for i, rc in enumerate(self.rate_controllers):
            new_learning_rate = rc.new_learning_rate(step=step, data=data)
            if new_learning_rate is None:
                return None
            else:
                result['group %d' % i] = new_learning_rate
        return result

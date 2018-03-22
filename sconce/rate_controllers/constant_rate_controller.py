from sconce.rate_controllers.base import RateController

from sconce.monitors.ringbuffer_monitor import RingbufferMonitor


class ConstantRateController(RateController):
    """
    A Learning rate that is constant (optionally stopping after
    some conditions are met).
    """
    def __init__(self, learning_rate, patience=None, key='training_loss',
            num_drops=0, drop_factor=0.1):
        self.learning_rate = learning_rate
        self.patience = patience
        self.key = key
        self.num_drops = num_drops
        self.drop_factor = drop_factor

        self.monitor = None
        self.num_drops_taken = 0
        self.factor = 1.0

    def start_session(self, *args):
        if self.patience is not None:
            self.reset_monitor()

    def reset_monitor(self):
        self.monitor = RingbufferMonitor(capacity=2 * self.patience,
                key=self.key)

    def new_learning_rate(self, step, data):
        if self.monitor is not None:
            self.monitor.step(data)
            is_moving = self.monitor.value_distribution_is_moving
            if is_moving is not None and not is_moving:
                if self.num_drops_taken < self.num_drops:
                    self.reset_monitor()
                    self.num_drops_taken += 1
                    self.factor *= self.drop_factor
                else:
                    return None

        return self.factor * self.learning_rate

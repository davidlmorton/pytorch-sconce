from sconce.rate_controllers.base import RateController

from sconce.monitors.ringbuffer_monitor import RingbufferMonitor


class ConstantRateController(RateController):
    """
    A Learning rate that is constant.  It can adjust its learning rate by
    <drop_factor> up to <num_drops> times based on detecting that some
    metric or loss has stopped moving.
    """
    def __init__(self, learning_rate,
            drop_factor=0.1,
            movement_key='training_loss',
            movement_threshold=0.25,
            movement_window=None,
            num_drops=0):
        self.learning_rate = learning_rate
        self.movement_window = movement_window
        self.movement_key = movement_key
        self.num_drops = num_drops
        self.drop_factor = drop_factor
        self.movement_threshold = movement_threshold

        self.monitor = None
        self.num_drops_taken = 0
        self.factor = 1.0

    def start_session(self, *args):
        if self.movement_window is not None:
            self.reset_monitor()

    def reset_monitor(self):
        self.monitor = RingbufferMonitor(capacity=2 * self.movement_window,
                key=self.movement_key)

    def new_learning_rate(self, step, data):
        if self.monitor is not None:
            self.monitor.write(data=data, step=step)
            movement_index = self.monitor.movement_index
            if movement_index is not None:
                if movement_index < self.movement_threshold:
                    if self.num_drops_taken < self.num_drops:
                        self.reset_monitor()
                        self.num_drops_taken += 1
                        self.factor *= self.drop_factor
                    else:
                        return None

        return self.factor * self.learning_rate

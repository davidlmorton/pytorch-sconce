from sconce.rate_controllers.base import RateController

import numpy as np


class StepRateController(RateController):
    """
    A Learning Rate that falls in <num_drops> drops from <max_learning_rate> to
    <min_learning_rate> over the course of <num_steps>.  The Learning Rate is
    constant between drops.
    """
    def __init__(self, max_learning_rate, min_learning_rate,
            num_drops=1):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

        self.num_regions = num_drops + 1
        self.region_values = np.linspace(max_learning_rate,
                min_learning_rate, self.num_regions)
        self.num_steps = None

    def start_session(self, num_steps):
        self.num_steps = num_steps
        self.idxs = [int(i) for i in
                np.linspace(0, self.num_regions - 1e-12,
                    self.num_steps)]

    def new_learning_rate(self, step, data):
        if self.num_steps is None:
            raise RuntimeError("You must call 'start_session' before calling "
                    "'new_learning_rate'")
        if step > self.num_steps:
            raise RuntimeError(f"Argument step={step}, should not "
                    f"exceed num_steps={self.num_steps}")

        return self.region_values[self.idxs[step - 1]]

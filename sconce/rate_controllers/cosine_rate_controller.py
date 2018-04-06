from sconce.rate_controllers.base import RateController

import math
import numpy as np


class CosineRateController(RateController):
    """
    A learning rate that follows a scaled and shifted cosine function
    from [0, pi/2].  It will begin at <max_learning_rate> and end at
    <min_learning_rate>, after <num_steps>.
    """
    def __init__(self, max_learning_rate, min_learning_rate=None):
        self.max_learning_rate = max_learning_rate

        if min_learning_rate is None:
            min_learning_rate = max_learning_rate / 50.0
        self.min_learning_rate = min_learning_rate

        self.num_steps = None

    def start_session(self, num_steps):
        self.num_steps = num_steps
        self.progressions = np.linspace(0, 1, num_steps)

    def new_learning_rate(self, step, data):
        if self.num_steps is None:
            raise RuntimeError("You must call 'start_session' before calling "
                    "'new_learning_rate'")
        if step > self.num_steps:
            raise RuntimeError(f"Argument step={step}, should not "
                    f"exceed num_steps={self.num_steps}")

        progression = self.progressions[step - 1]

        new_learning_rate = (self.min_learning_rate +
                (self.max_learning_rate - self.min_learning_rate) *
                (1 + math.cos(math.pi * progression)) / 2)
        return new_learning_rate

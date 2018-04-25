from sconce.monitors.base import Monitor
from torch import Tensor

import math
import numpy as np
from numpy_ringbuffer import RingBuffer


class RingbufferMonitor(Monitor):
    def __init__(self, capacity=100,
            key='training_loss',
            name='ringbuffer_monitor'):
        super().__init__(name=name)
        self.capacity = capacity
        self.key = key
        self.value_buffer = None
        self.step_buffer = RingBuffer(capacity=capacity, dtype='float32')

        self.previous_session_steps = 0
        self.last_step = 0

    def start_session(self, num_steps, **kwargs):
        self.previous_session_steps += self.last_step

    def write(self, data, step, **kwargs):
        if self.key in data.keys():
            value = data[self.key]

            if isinstance(value, Tensor):
                dtype = value.data.cpu().numpy().dtype
                value = self._to_scalar(value)
            else:
                dtype = np.array(value).dtype

            if self.value_buffer is None:
                self.value_buffer = RingBuffer(capacity=self.capacity,
                        dtype=dtype)

            self.value_buffer.append(value)
            self.step_buffer.append(step + self.previous_session_steps)

        self.last_step = math.ceil(step)

    def mean(self, *args, **kwargs):
        return np.array(self.value_buffer).mean(*args, **kwargs)

    def std(self, *args, **kwargs):
        return np.array(self.value_buffer).std(*args, **kwargs)

    @property
    def movement_index(self):
        steps = self.step_buffer
        if len(steps) < self.capacity:
            # not enough information gathered yet to say...
            return None
        else:
            step_difference = (steps[-1] - steps[0])
            midpoint = np.argmax(steps >=
                    steps[0] + (step_difference // 2))

            values = np.array(self.value_buffer)

            first = values[:midpoint]
            last = values[midpoint:]

            d = (np.abs(first.mean() - last.mean()) /
                    (2 * np.sqrt(first.std() * last.std())))
            return d

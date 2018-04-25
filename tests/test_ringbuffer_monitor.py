from sconce.monitors import RingbufferMonitor
import unittest
import numpy as np


class TestRingbufferMonitor(unittest.TestCase):
    def test_is_moving_true(self):
        r1 = np.random.randn(1000) + 50
        r2 = np.random.randn(1000) + 40

        rb = RingbufferMonitor(capacity=(len(r1) + len(r2)))
        for i, r in enumerate(np.concatenate([r1, r2])):
            # always returns None until buffer is filled up
            self.assertIs(None, rb.movement_index)

            rb.write(data={'training_loss': r}, step=i)

        self.assertGreater(rb.movement_index, 2)

    def test_is_moving_false(self):
        r1 = np.random.randn(1000) + 50
        r2 = np.random.randn(1000) + 50

        rb = RingbufferMonitor(capacity=(len(r1) + len(r2)))
        for i, r in enumerate(np.concatenate([r1, r2])):
            rb.write(data={'training_loss': r}, step=i)

        self.assertLess(rb.movement_index, 0.2)

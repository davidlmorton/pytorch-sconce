from sconce.monitors import RingbufferMonitor
import unittest
import numpy as np


class TestRingbufferMonitor(unittest.TestCase):
    def test_is_moving_true(self):
        r1 = np.random.randn(1000) + 50
        r2 = np.random.randn(1000) + 40

        rb = RingbufferMonitor(capacity=(len(r1) + len(r2)))
        for r in np.concatenate([r1, r2]):
            # always returns None until buffer is filled up
            self.assertIs(None, rb.movement_index)

            rb.step(data={'training_loss': r})

        self.assertTrue(rb.movement_index > 2)

    def test_is_moving_false(self):
        r1 = np.random.randn(1000) + 50
        r2 = np.random.randn(1000) + 50

        rb = RingbufferMonitor(capacity=(len(r1) + len(r2)))
        for r in np.concatenate([r1, r2]):
            rb.step(data={'training_loss': r})

        self.assertTrue(rb.movement_index < 0.2)

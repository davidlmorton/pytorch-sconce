from sconce import rate_controllers as rc
import unittest


class TestRateControllers(unittest.TestCase):
    def test_cosine(self):
        crc = rc.CosineRateController(
                max_learning_rate=1,
                min_learning_rate=0)

        with self.assertRaises(RuntimeError):
            crc.new_learning_rate(step=0, data={})

        crc.start_session(num_steps=3)

        with self.assertRaises(RuntimeError):
            crc.new_learning_rate(step=3, data={})

        expected_values = (1, 0.5, 0)
        for step, expected_value in enumerate(expected_values):
            value = crc.new_learning_rate(step=step, data={})
            self.assertAlmostEqual(value, expected_value)

    def test_exponential(self):
        erc = rc.ExponentialRateController(
                min_learning_rate=0.1,
                max_learning_rate=1)

        with self.assertRaises(RuntimeError):
            erc.new_learning_rate(step=0, data={})

        erc.start_session(num_steps=3)

        with self.assertRaises(RuntimeError):
            erc.new_learning_rate(step=3, data={})

        expected_values = (0.1, 0.31622777, 1)
        for step, expected_value in enumerate(expected_values):
            value = erc.new_learning_rate(step=step, data={})
            self.assertAlmostEqual(value, expected_value)

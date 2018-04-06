from sconce import rate_controllers as rc
import unittest


class TestRateControllers(unittest.TestCase):
    def test_cosine(self):
        crc = rc.CosineRateController(
                max_learning_rate=1,
                min_learning_rate=0)

        with self.assertRaises(RuntimeError):
            crc.new_learning_rate(step=1, data={})

        crc.start_session(num_steps=3)

        with self.assertRaises(RuntimeError):
            crc.new_learning_rate(step=4, data={})

        expected_values = (1, 0.5, 0)
        for i, expected_value in enumerate(expected_values):
            value = crc.new_learning_rate(step=i + 1, data={})
            self.assertAlmostEqual(value, expected_value)

    def test_exponential(self):
        erc = rc.ExponentialRateController(
                min_learning_rate=0.1,
                max_learning_rate=1)

        with self.assertRaises(RuntimeError):
            erc.new_learning_rate(step=1, data={})

        erc.start_session(num_steps=3)

        with self.assertRaises(RuntimeError):
            erc.new_learning_rate(step=4, data={})

        expected_values = (0.1, 0.31622777, 1)
        for i, expected_value in enumerate(expected_values):
            value = erc.new_learning_rate(step=i + 1, data={})
            self.assertAlmostEqual(value, expected_value)

    def test_step(self):
        controller = rc.StepRateController(
                max_learning_rate=1.5,
                min_learning_rate=0.5,
                num_drops=2)

        with self.assertRaises(RuntimeError):
            controller.new_learning_rate(step=0, data={})

        expected_values_list = [
            (1.5, 1.0, 0.5),
            (1.5, 1.5, 1.0, 0.5),
            (1.5, 1.5, 1.0, 0.5, 0.5),
            (1.5, 1.5, 1.0, 1.0, 0.5, 0.5),
            (1.5, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5),
            (1.5, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5),
        ]
        for expected_values in expected_values_list:
            self._controller_test(controller, expected_values)

    def test_triangle(self):
        controller = rc.TriangleRateController(
                max_learning_rate=0.2,
                min_learning_rate=0.1)

        with self.assertRaises(RuntimeError):
            controller.new_learning_rate(step=0, data={})

        expected_values_list = [
            (0.1, 0.2, 0.15, 0.1),
            (0.1, 0.15, 0.2, 0.15, 0.1),
            (0.1, 0.15, 0.2, 0.167, 0.133, 0.1),
            (0.1, 0.133, 0.167, 0.2, 0.167, 0.133, 0.1),
        ]
        for expected_values in expected_values_list:
            self._controller_test(controller, expected_values)

    def _controller_test(self, controller, expected_values):
        print(f'Expecting: {expected_values}')
        num_steps = len(expected_values)
        controller.start_session(num_steps=num_steps)

        with self.assertRaises(RuntimeError):
            controller.new_learning_rate(step=num_steps + 1, data={})

        got = []
        for i, expected_value in enumerate(expected_values):
            value = controller.new_learning_rate(step=i + 1, data={})
            got.append(value)
        print(f'Got: {got}')

        for i in range(len(got)):
            self.assertAlmostEqual(got[i], expected_values[i], 3)

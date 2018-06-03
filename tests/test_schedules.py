from sconce import schedules
import unittest


class TestSchedules(unittest.TestCase):
    def test_num_steps(self):
        s = schedules.Cosine(initial_value=0, final_value=1)
        s.set_num_steps(3)

        with self.assertRaises(RuntimeError):
            s.get_value(step=4, current_state={})

    def test_cosine(self):
        s = schedules.Cosine(initial_value=0, final_value=1)
        self.assertEqual('Cosine(initial_value=0, final_value=1)', str(s))
        s.set_num_steps(3)

        expected_values = (0, 0.5, 1)
        for i, expected_value in enumerate(expected_values):
            result = s.get_value(step=i + 1, current_state={})
            self.assertAlmostEqual(result, expected_value)

    def test_exponential(self):
        s = schedules.Exponential(initial_value=0.1, final_value=1)
        self.assertEqual(
                "Exponential(initial_value=0.1, final_value=1, stop_factor=None, loss_key='training_loss')", str(s))
        s.set_num_steps(3)

        expected_values = (0.1, 0.31622777, 1)
        for i, expected_value in enumerate(expected_values):
            result = s.get_value(step=i + 1, current_state={})
            self.assertAlmostEqual(result, expected_value)

    def test_step(self):
        expected_values_list = [
            (1.5, 1.0, 0.5),
            (1.5, 1.5, 1.0, 0.5),
            (1.5, 1.5, 1.0, 0.5, 0.5),
            (1.5, 1.5, 1.0, 1.0, 0.5, 0.5),
            (1.5, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5),
            (1.5, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5),
        ]
        s = schedules.Step(initial_value=1.5, final_value=0.5, num_changes=2)
        self.assertEqual('Step(initial_value=1.5, final_value=0.5, num_changes=2)', str(s))
        for expected_values in expected_values_list:
            self._schedule_test(expected_values, s)

    def _schedule_test(self, expected_values, s):
        print(f'Expecting: {expected_values}')
        num_steps = len(expected_values)
        s.set_num_steps(num_steps)

        got = []
        for i, expected_value in enumerate(expected_values):
            value = s.get_value(step=i + 1, current_state={})
            got.append(value)
        print(f'Got: {got}')

        for i in range(len(got)):
            self.assertAlmostEqual(got[i], expected_values[i], 3)

    def test_triangle(self):
        expected_values_list = [
            (0.1, 0.2, 0.15, 0.1),
            (0.1, 0.15, 0.2, 0.15, 0.1),
            (0.1, 0.15, 0.2, 0.167, 0.133, 0.1),
            (0.1, 0.133, 0.167, 0.2, 0.167, 0.133, 0.1),
        ]
        s = schedules.Triangle(initial_value=0.1, peak_value=0.2)
        self.assertEqual('Triangle(initial_value=0.1, peak_value=0.2, peak_fraction=0.5)', str(s))
        for expected_values in expected_values_list:
            self._schedule_test(expected_values, s)

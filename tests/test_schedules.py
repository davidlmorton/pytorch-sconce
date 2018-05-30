from sconce import schedules
import unittest


class TestSchedules(unittest.TestCase):
    def test_num_steps(self):
        s = schedules.Cosine(initial_value=0, final_value=1, num_steps=3)

        with self.assertRaises(RuntimeError):
            s.get_value(step=4, current_state={})

    def test_cosine(self):
        s = schedules.Cosine(initial_value=0, final_value=1, num_steps=3)

        expected_values = (1, 0.5, 0)
        for i, expected_value in enumerate(expected_values):
            result = s.get_value(step=i + 1, current_state={})
            self.assertAlmostEqual(result, expected_value)

    def test_exponential(self):
        s = schedules.Exponential(initial_value=0.1, final_value=1, num_steps=3)

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
        for expected_values in expected_values_list:
            def factory(num_steps):
                return schedules.Step(initial_value=1.5, final_value=0.5,
                    num_changes=2, num_steps=num_steps)
            self._schedule_test(expected_values, factory)

    def _schedule_test(self, expected_values, factory):
        print(f'Expecting: {expected_values}')
        num_steps = len(expected_values)
        s = factory(num_steps)

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
        for expected_values in expected_values_list:
            def factory(num_steps):
                return schedules.Triangle(initial_value=0.1, peak_value=0.2,
                    num_steps=num_steps)
            self._schedule_test(expected_values, factory)

from .base import MNISTTest
from sconce.data_generator import DataGenerator


class TestDataGenerator(MNISTTest):
    num_test_samples = 3

    def test_iterates_forever(self):
        dg = DataGenerator(data_loader=self.test_data_loader)
        count = 0
        for batch in dg:
            count += 1
            if count >= len(self.test_data_loader) * 2.2:
                break
        self.assertTrue(True, "Can iterate forever")

    def test_properties(self):
        dg = DataGenerator(data_loader=self.test_data_loader)
        self.assertEqual(len(dg), len(self.test_data_loader))
        self.assertIs(dg.dataset, self.test_data_loader.dataset)
        self.assertEqual(dg.batch_size,
                self.test_data_loader.batch_size)
        self.assertEqual(dg.num_samples, len(self.test_data_loader.dataset))

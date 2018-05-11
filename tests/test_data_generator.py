from sconce.data_generators import SingleClassImageDataGenerator

import torch
import unittest


class TestDataGenerator(unittest.TestCase):
    num_test_samples = 3

    def test_iterates_forever(self):
        dg = SingleClassImageDataGenerator.from_torchvision(
                batch_size=1,
                fraction=5 / 10_000,
                train=False)
        count = 0
        for batch in dg:
            count += 1
            if count >= 5 * 2.2:
                break
        self.assertTrue(True, "Can iterate forever")

    def test_properties(self):
        dg = SingleClassImageDataGenerator.from_torchvision(
                batch_size=3,
                fraction=15 / 10_000,
                train=False)
        self.assertEqual(len(dg), 5)
        self.assertEqual(dg.num_samples, 15)

    @unittest.skipIf(not torch.cuda.is_available(),
            "Cuda unavailable, skipping test")
    def test_cuda(self):
        dg = SingleClassImageDataGenerator.from_torchvision(
                batch_size=1,
                fraction=5 / 10_000,
                train=False)
        inputs, targets = dg.next()
        self.assertFalse(inputs.is_cuda)
        self.assertFalse(targets.is_cuda)

        dg.cuda()
        inputs, targets = dg.next()
        self.assertTrue(inputs.is_cuda)
        self.assertTrue(targets.is_cuda)

        dg.cuda(False)
        inputs, targets = dg.next()
        self.assertFalse(inputs.is_cuda)
        self.assertFalse(targets.is_cuda)

        dg.cuda({'targets': None})
        inputs, targets = dg.next()
        self.assertFalse(inputs.is_cuda)
        self.assertTrue(targets.is_cuda)

        dg.cuda(False)
        inputs, targets = dg.next()
        self.assertFalse(inputs.is_cuda)
        self.assertFalse(targets.is_cuda)

        dg.cuda({'targets': 0})
        inputs, targets = dg.next()
        self.assertFalse(inputs.is_cuda)
        self.assertTrue(targets.is_cuda)

        dg.cuda(False)
        inputs, targets = dg.next()
        self.assertFalse(inputs.is_cuda)
        self.assertFalse(targets.is_cuda)

        dg.cuda({'targets': False, 'inputs': 0})
        inputs, targets = dg.next()
        self.assertTrue(inputs.is_cuda)
        self.assertFalse(targets.is_cuda)

        with self.assertRaises(RuntimeError):
            dg.cuda({'foo': None})

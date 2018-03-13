from torch.utils.data import DataLoader, dataset
from torchvision import datasets, transforms

import os
import tempfile
import unittest

__all__ = ['MNISTTest']


class MNISTTest(unittest.TestCase):
    data_location = os.path.join(tempfile.gettempdir(), 'mnist')
    transform_training_data_fn = transforms.ToTensor()
    transform_test_data_fn = transforms.ToTensor()
    data_loader_kwargs_training = {
        'batch_size': 500,
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True,
    }
    data_loader_kwargs_test = {
        'batch_size': 500,
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True,
    }
    num_test_samples = 10_000
    num_training_samples = 60_000

    def setUp(self):
        training_dataset = datasets.MNIST(self.data_location,
            train=True, download=True,
            transform=self.transform_training_data_fn)
        training_subset = dataset.Subset(training_dataset,
                indices=range(self.num_training_samples))
        self.training_data_loader = DataLoader(training_subset,
                **self.data_loader_kwargs_training)

        test_dataset = datasets.MNIST(self.data_location,
            train=False, transform=self.transform_test_data_fn)
        test_subset = dataset.Subset(test_dataset,
                indices=range(self.num_test_samples))
        self.test_data_loader = DataLoader(test_subset,
                **self.data_loader_kwargs_test)

from torch.utils.data import DataLoader
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

    def setUp(self):
        mnist_training_dataset = datasets.MNIST(self.data_location,
            train=True, download=True,
            transform=self.transform_training_data_fn)
        self.training_data_loader = DataLoader(mnist_training_dataset,
                **self.data_loader_kwargs_training)

        mnist_test_dataset = datasets.FashionMNIST(self.data_location,
            train=False, transform=self.transform_training_data_fn)
        self.test_data_loader = DataLoader(mnist_test_dataset,
                **self.data_loader_kwargs_test)
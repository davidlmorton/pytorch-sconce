from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms

import os
import tempfile
import numpy as np


class DataGenerator:
    """
    A thin wrapper around a <data_loader> that turns torch.Tensors into
    torch.Variables (that live on cpu or on cuda) and iterates endlessly.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._inputs_cuda = False
        self._targets_cuda = False
        self.reset()

    def cuda(self, device=None):
        if isinstance(device, dict):
            for key, value in device.items():
                if key == 'inputs':
                    self._inputs_cuda = value
                elif key == 'targets':
                    self._targets_cuda = value
                else:
                    raise RuntimeError(f"Invalid key for 'device' argument: "
                            f"({key}) expected to be in ('inputs', 'targets').")
        else:
            self._inputs_cuda = device
            self._targets_cuda = device

    def reset(self):
        self._iterator = iter(self.data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    @property
    def batch_size(self):
        return self.data_loader.batch_size

    @property
    def num_samples(self):
        return len(self.dataset)

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def preprocess(self, inputs, targets):
        inputs = Variable(inputs)
        targets = Variable(targets)

        if self._inputs_cuda is False:
            inputs = inputs.cpu()
        else:
            inputs = inputs.cuda(self._inputs_cuda)

        if self._targets_cuda is False:
            targets = targets.cpu()
        else:
            targets = targets.cuda(self._targets_cuda)

        return inputs, targets

    def next(self):
        try:
            inputs, targets = self._iterator.next()
        except StopIteration:
            self.reset()
            inputs, targets = self._iterator.next()
        return self.preprocess(inputs, targets)

    @classmethod
    def from_pytorch(cls,
            batch_size=500,
            data_location=None,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            train=True,
            fraction=1.0,
            transform=transforms.ToTensor(),
            dataset_class=datasets.MNIST):
        assert(fraction > 0.0)
        assert(fraction <= 1.0)

        if data_location is None:
            data_location = os.path.join(tempfile.gettempdir(),
                    dataset_class.__name__)

        dataset = dataset_class(data_location,
                train=train,
                download=True,
                transform=transform)
        indices = [int(x) for x in np.linspace(
                start=0,
                stop=len(dataset) - 1,
                num=int(len(dataset) * fraction))]
        subset = data.dataset.Subset(dataset, indices=indices)
        return cls.from_dataset(subset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        data_loader = data.DataLoader(dataset, **kwargs)
        return cls(data_loader)

from torch.utils import data
from torchvision import datasets, transforms

import os
import tempfile
import numpy as np


class DataGenerator:
    """
    A thin wrapper around a :py:class:`~torch.utils.data.DataLoader` that
    automatically yields tuples of :py:class:`torch.Tensor` (that
    live on cpu or on cuda).
    A DataGenerator will iterate endlessly.

    Like the underlying :py:class:`~torch.utils.data.DataLoader`, a
    DataGenerator's ``__next__`` method yields two values, which we refer to as
    the `inputs` and the `targets`.

    Arguments:
        data_loader (:py:class:`~torch.utils.data.DataLoader`): the wrapped
            data_loader.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._inputs_cuda = False
        self._targets_cuda = False
        self.reset()

    def cuda(self, device=None):
        """
        Put the `inputs` and `targets` (yielded by this DataGenerator) on the
        specified device.

        Arguments:
            device (int or bool or dict): if int or bool, sets the behavior for
                both `inputs` and `targets`. To set them individually, pass a
                dictionary with keys {'inputs', 'targets'} instead. See
                :py:meth:`torch.Tensor.cuda` for details.

        Example:
            >>> g = DataGenerator.from_dataset(dataset, batch_size=100)
            >>> g.cuda()
            >>> g.next()
            (Tensor containing:
             [torch.cuda.FloatTensor of size 100x1x28x28 (GPU 0)],
             Tensor containing:
             [torch.cuda.LongTensor of size 100 (GPU 0)])
            >>> g.cuda(False)
            >>> g.next()
            (Tensor containing:
             [torch.FloatTensor of size 100x1x28x28],
             Tensor containing:
             [torch.LongTensor of size 100])
            >>> g.cuda(device={'inputs':0, 'targets':1})
            >>> g.next()
            (Tensor containing:
             [torch.cuda.FloatTensor of size 100x1x28x28 (GPU 0)],
             Tensor containing:
             [torch.cuda.LongTensor of size 100 (GPU 1)])
        """
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
        """
        Start iterating through the data_loader from the begining.
        """
        self._iterator = iter(self.data_loader)

    @property
    def dataset(self):
        """
        the wrapped data_loader's :py:class:`~torch.utils.data.Dataset`
        """
        return self.data_loader.dataset

    @property
    def real_dataset(self):
        """
        the wrapped data_loader's :py:class:`~torch.utils.data.Dataset` reaching through any Subsets
        """
        return self._real_dataset(self.dataset)

    def _real_dataset(self, dataset):
        result = dataset
        if isinstance(result, data.dataset.Subset):
            return self._real_dataset(result.dataset)
        else:
            return result

    @property
    def batch_size(self):
        """
        the wrapped data_loader's batch_size
        """
        return self.data_loader.batch_size

    @property
    def num_samples(self):
        """
        the ``len`` of the wrapped data_loader's
        :py:class:`~torch.utils.data.Dataset`
        """
        return len(self.dataset)

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def preprocess(self, inputs, targets):
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
            dataset_class=datasets.MNIST,
            fraction=1.0,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            train=True,
            transform=transforms.ToTensor()):
        """
        Note:
            This method is deprecated as of 0.8.0, and will be removed in 0.9.0.

        Create a DataGenerator from a torchvision dataset class.

        Arguments:
            batch_size (int): how large the yielded `inputs` and `targets`
                should be. See :py:class:`DataLoader` for details.
            data_location (path): where downloaded dataset should be stored.  If
                ``None`` a system dependent temporary location will be used.
            dataset_class (class): a torchvision dataset class that supports
                constructor arguments {'root', 'train', 'download',
                'transform'}. For example, MNIST, FashionMnist, CIFAR10, or
                CIFAR100.
            fraction (float): (0.0 - 1.0] how much of the original dataset's
                data to use.
            num_workers (int): how many subprocesses to use for data loading.
                See :py:class:`DataLoader` for details.
            pin_memory (bool): if ``True``, the data loader will copy tensors
                into CUDA pinned memory before returning them. See
                :py:class:`DataLoader` for details.
            shuffle (bool): set to ``True`` to have the data reshuffled at every
                epoch. See :py:class:`DataLoader` for details.
            train (bool): if ``True``, creates dataset from training set,
                otherwise creates from test set.
            transform (callable): a function/transform that takes in an PIL
                image and returns a transformed version.
        """
        print("WARNING: DataGenerator.from_pytorch method has been deprecated.  "
                "Please use ImageDataGenerator.from_torchvision instead.")
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
        """
        Create a DataGenerator from an instantiated dataset.

        Arguments:
            dataset (:py:class:`~torch.utils.data.Dataset`): the pytorch
                dataset.
            **kwargs: passed directly to the
                :py:class:`~torch.utils.data.DataLoader`) constructor.
        """
        data_loader = data.DataLoader(dataset, **kwargs)
        return cls(data_loader)

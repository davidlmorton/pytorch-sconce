from sconce.data_generators import DataGenerator, ImageMixin
from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import os
import pandas as pd
import tempfile


class SingleClassImageDataGenerator(DataGenerator, ImageMixin):
    """
    An ImageDataGenerator class for use when each image belongs to exactly one class.

    New in 0.10.0
    """
    def _get_class_df(self):
        dataset = self.real_dataset
        rows = []

        for target in dataset.targets:
            row = {}
            for _class in dataset.classes:
                idx = dataset.class_to_idx[_class]
                if target == idx:
                    row[_class] = True
                else:
                    row[_class] = False
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def from_torchvision(cls,
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
    def from_image_folder(self, root, loader_kwargs=None, **dataset_kwargs):
        """
        Create a DataGenerator from a folder of images.  See :py:class:`torchvision.datasets.ImageFolder`.

        Arguments:
            root (path): the root directory path.
            loader_kwargs (dict): keyword args provided to the DataLoader constructor.
            **dataset_kwargs: keyword args provided to the :py:class:`torchvision.datasets.ImageFolder` constructor.
        """
        if loader_kwargs is None:
            loader_kwargs = {}

        dataset = datasets.ImageFolder(root=root, **dataset_kwargs)
        return self.from_dataset(dataset, **loader_kwargs)

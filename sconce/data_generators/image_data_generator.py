from .base import DataGenerator
from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import os
import pandas as pd
import seaborn as sns
import tempfile


def get_image_info(image):
    return {'height': image.height, 'width': image.width, 'num_channels': len(image.getbands())}


class ImageDataGenerator(DataGenerator):
    """
    A DataGenerator class with some handy methods for image type data.

    New in 0.7.0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.summary_df = None

    @property
    def num_channels(self):
        """
        The number of image channels, based on looking at the first image in the dataset.
        """
        dataset = self.real_dataset
        for image, target in dataset:
            return image.shape[0]

    def get_summary_df(self):
        """
        Return a pandas dataframe that summarizes the image metadata in the dataset.
        """
        if self.summary_df is None:
            self.summary_df = self._get_summary_df()
        return self.summary_df

    def _get_summary_df(self):
        info_list = []

        dataset = self.real_dataset

        old_transform = self.real_dataset.transform
        old_target_tranform = self.real_dataset.target_transform
        try:
            dataset.transform = None
            dataset.target_transform = None
            for image, label in dataset:
                info = get_image_info(image)
                info['label'] = label
                info_list.append(info)
        except Exception:
            pass

        dataset.transform = old_transform
        dataset.target_transform = old_target_tranform

        return pd.DataFrame(info_list)

    def plot_label_summary(self):
        """
        Generate a barchart showing how many images of each label there are.
        """
        summary_df = self.get_summary_df()
        return sns.countplot(x='label', data=summary_df)

    def plot_size_summary(self):
        """
        Generate a scatter plot showing the sizes of the images in the dataset.
        """
        summary_df = self.get_summary_df()
        return sns.jointplot(x="height", y="width",
                kind='scatter', stat_func=None, data=summary_df)

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

from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns


def get_image_size(image):
    return {'height': image.height, 'width': image.width}


class ImageMixin(ABC):
    """
    A DataGenerator Mixin class with some handy methods for image type data.

    New in 0.10.0 (used to be called ImageDataGenerator and not be abstract)
    """
    @property
    def num_channels(self):
        """
        The number of image channels, based on looking at the first image in the dataset.
        """
        dataset = self.real_dataset
        for image, target in dataset:
            return image.shape[0]

    def get_image_size_df(self):
        """
        Return a pandas dataframe that contains the image sizes in the dataset (before transforms).
        """
        if not hasattr(self, '_image_size_df'):
            self._image_size_df = self._get_image_size_df()
        return self._image_size_df

    def _get_image_size_df(self):
        info_list = []

        dataset = self.real_dataset

        old_transform = dataset.transform
        try:
            dataset.transform = None
            dataset.target_transform = None
            for image, _ in dataset:
                info = get_image_size(image)
                info_list.append(info)
        except Exception as e:
            print(e)
            pass
        dataset.transform = old_transform

        return pd.DataFrame(info_list)

    def get_class_df(self):
        """
        Return a pandas dataframe that contains the classes in the dataset.
        """
        if not hasattr(self, '_class_df'):
            self._class_df = self._get_class_df()
        return self._class_df

    @abstractmethod
    def _get_class_df(self):
        pass

    def plot_class_summary(self, **kwargs):
        """
        Generate a barchart showing how many images of each class there are.
        """
        df = self.get_class_df()
        plot_kwargs = {'kind': 'bar', **kwargs}
        return df.sum().plot(**plot_kwargs)

    def plot_image_size_summary(self):
        """
        Generate a scatter plot showing the sizes of the images in the dataset.
        """
        df = self.get_image_size_df()
        return sns.jointplot(x="height", y="width",
                kind='scatter', stat_func=None, data=df)

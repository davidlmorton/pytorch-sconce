from matplotlib import pyplot as plt
from sconce.data_feeds.base import DataFeed
from torchvision import transforms

import pandas as pd
import seaborn as sns


class ImageFeed(DataFeed):
    """
    A Datafeed with some handy methods for image type data.
    """
    @property
    def num_channels(self):
        """
        The number of image channels, based on looking at the first image in the dataset.
        """
        dataset = self.dataset
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

        dataset = self.dataset

        old_transform = dataset.transform
        try:
            dataset.transform = None
            for image, _ in dataset:
                info = {'height': image.height, 'width': image.width}
                info_list.append(info)
        except Exception as e:
            print(e)
            pass
        dataset.transform = old_transform

        return pd.DataFrame(info_list)

    def reset_class_df_cache(self):
        del self._class_df

    def get_class_df(self):
        """
        Return a pandas dataframe that contains the classes in the dataset.
        """
        if not hasattr(self, '_class_df'):
            self._class_df = self._get_class_df()
        return self._class_df

    def _get_class_df(self):
        dataset = self.dataset

        rows = [{'unclassified': True} for i in range(len(dataset))]
        return pd.DataFrame(rows)

    def plot_class_summary(self, **kwargs):
        """
        Generate a barchart showing how many images of each class there are.
        """
        df = self.get_class_df()
        plot_kwargs = {'kind': 'bar', **kwargs}
        return df.sum().plot(**plot_kwargs)

    def reset_targets_cache(self):
        del self._targets

    def get_targets(self):
        if not hasattr(self, '_targets'):
            self._targets = self._get_targets()
        return self._targets

    def _get_targets(self):
        if hasattr(self.dataset, 'targets'):
            return self.dataset.targets
        else:
            return [item[1] for item in self.dataset]

    def plot_image_size_summary(self):
        """
        Generate a scatter plot showing the sizes of the images in the dataset.
        """
        df = self.get_image_size_df()
        return sns.jointplot(x="height", y="width",
                kind='scatter', stat_func=None, data=df)

    def _get_original_image(self, index):
        dataset = self.dataset
        old_transform = dataset.transform

        dataset.transform = transforms.ToTensor()
        image = self.dataset[index][0]

        dataset.transform = old_transform
        return image

    def plot_transforms(self, index,
            num_samples=5,
            num_cols=5,
            figure_width=15,
            image_height=3,
            return_fig=False):
        """
        Plot the same image from this Datafeed multiple times to see how the transforms affect them.

        Arguments:
            index (int): the index of the image to plot.
            num_samples (int, optional): the number of times to plot the image (1 original, n - 1 transformed
                variations).
            num_cols (int, optional): the number of columns in the plot grid.
            num_cols (int): the number of columns to plot, one image per column.
            figure_width (float): the size, in matplotlib-inches, for the width of the whole figure.
            image_height (float): the size, in matplotlib-inches, for the height of a single image.
            return_fig (bool): return the generated matplotlib figure or not.
        """

        original = self._get_original_image(index)
        samples = [original] + [self.dataset[index][0] for i in range(num_samples - 1)]

        num_rows = -(-num_samples // num_cols)
        fig = plt.figure(figsize=(figure_width, image_height * num_rows))

        for i in range(num_samples):
            image = samples[i].cpu().data.numpy()

            if image.shape[0] == 1:
                # greyscale image
                image = image[0]
                cmap = 'gray'
            else:
                # color channels present
                image = image.swapaxes(0, 2)
                image = image.swapaxes(0, 1)
                cmap = None

            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.imshow(image, cmap=cmap)
            if i != 0:
                if i != 1:
                    ax.axis('off')
                ax.set_title('transformed')
            else:
                ax.set_title('original')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05)

        if return_fig:
            return fig

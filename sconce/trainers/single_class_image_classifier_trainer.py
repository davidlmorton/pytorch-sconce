from abc import ABC
from scipy import sparse
from sconce.trainer import Trainer
from matplotlib import pyplot as plt

import seaborn as sn
import numpy as np


__all__ = ['ClassifierTrainer', 'SingleClassImageClassifierMixin', 'SingleClassImageClassifierTrainer']


class SingleClassImageClassifierMixin(ABC):
    def get_confusion_matrix(self, data_generator=None, cache_results=True):
        if data_generator is None:
            data_generator = self.test_data_generator

        run_model_results = self._run_model_on_generator(data_generator,
                cache_results=cache_results)

        targets = run_model_results['targets']
        predicted_targets = np.argmax(run_model_results['outputs'], axis=1)
        matrix = sparse.coo_matrix((np.ones(len(targets)),
                (predicted_targets, targets)), dtype='uint32').toarray()
        return matrix

    def get_classification_accuracy(self, data_generator=None,
            cache_results=True):
        if data_generator is None:
            data_generator = self.test_data_generator

        matrix = self.get_confusion_matrix(data_generator=data_generator,
                cache_results=cache_results)
        num_correct = np.trace(matrix)
        return num_correct / data_generator.num_samples

    def plot_confusion_matrix(self, data_generator=None, **heatmap_kwargs):
        if data_generator is None:
            data_generator = self.test_data_generator

        matrix = self.get_confusion_matrix(data_generator=data_generator)

        dataset = data_generator.real_dataset
        defaults = {'cmap': 'YlGnBu', 'annot': True, 'fmt': 'd',
                    'xticklabels': dataset.classes,
                    'yticklabels': dataset.classes}
        ax = sn.heatmap(matrix, **{**defaults, **heatmap_kwargs})

        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0)
        ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(),
                rotation=0, ha='right')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        return ax

    def plot_samples(self, predicted_class,
            true_class=None,
            data_generator=None,
            sort_by='rising predicted class score',
            num_samples=7,
            num_cols=7,
            figure_width=15,
            image_height=3,
            cache_results=True):
        """
        Plot samples of the dataset where the given <predicted_class> was predicted by the model.

        Arguments:
            predicted_class (int or string): the class string or the index of the class that was predicted by the model.
            true_class (int or string): the class string or the index of the class that the image actually belongs to.
            data_generator (:py:class:`~sconce.data_generators.base.SingleClassImageDataGenerator`): the data generator
                to use to find the samples.
            sort_by (string): one of the sort_by strings, see note below.
            num_samples (int): the number of sample images to plot.
            num_cols (int): the number of columns to plot, one image per column.
            figure_width (float): the size, in matplotlib-inches, for the width of the whole figure.
            image_height (float): the size, in matplotlib-inches, for the height of a single image.
            cache_results (bool): keep the results in memory to make subsequent plots faster.  Beware, that on large
                datasets (like imagenet) this can cause your system to run out of memory.

        Note:
            The sort_by strings supported are:
                - "rising predicted class score": samples are plotted in order of the lowest predicted class score to
                  the highest predicted class score.
                - "falling predicted class score": samples are plotted in order of the higest predicted class score to
                  the lowest predicted class score.
                - "rising true class score": samples are plotted in order of the lowest true class score to
                  the highest true class score.
                - "falling true class score": samples are plotted in order of the higest true class score to
                  the lowest true class score.
        """
        if data_generator is None:
            data_generator = self.test_data_generator

        dataset = data_generator.real_dataset
        predicted_class = self._convert_to_class_index(predicted_class, dataset)
        true_class = self._convert_to_class_index(true_class, dataset, default=predicted_class)

        run_model_results = self._run_model_on_generator(data_generator,
                cache_results=cache_results)

        images = run_model_results['inputs']
        targets = run_model_results['targets']
        outputs = run_model_results['outputs']

        predicted_targets = np.argmax(outputs, axis=1)
        keep_idxs = ((targets == true_class) &
                     (predicted_targets == predicted_class))
        kept_images = images[keep_idxs]
        predicted_class_scores = np.exp(outputs[keep_idxs, predicted_class])
        true_class_scores = np.exp(outputs[keep_idxs, true_class])

        kept_images = np.array(kept_images)
        predicted_class_scores = np.array(predicted_class_scores)
        true_class_scores = np.array(true_class_scores)

        sort_fns = {
            'rising predicted class score': lambda p, t: np.argsort(p),
            'falling predicted class score': lambda p, t: np.argsort(p)[::-1],
            'rising true class score': lambda p, t: np.argsort(t),
            'falling true class score': lambda p, t: np.argsort(t)[::-1],
        }

        sort_fn = sort_fns[sort_by]
        sort_key = sort_fn(predicted_class_scores, true_class_scores)
        sorted_kept_images = kept_images[sort_key]
        sorted_predicted_class_scores = predicted_class_scores[sort_key]
        sorted_true_class_scores = true_class_scores[sort_key]

        if num_samples < len(kept_images):
            print(f'Showing only the first {num_samples} of '
                  f'{len(kept_images)} images')

        num_samples = min(num_samples, len(kept_images))
        num_rows = -(-num_samples // num_cols)
        fig = plt.figure(figsize=(figure_width, image_height * num_rows))

        for i in range(num_samples):
            image = sorted_kept_images[i]
            predicted_class_score = sorted_predicted_class_scores[i]
            true_class_score = sorted_true_class_scores[i]

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
            if true_class != predicted_class:
                ax.set_title('p: %2.1f%%\nt: %2.1f%%' % (
                    predicted_class_score * 100, true_class_score * 100))
            else:
                ax.set_title('%2.1f%%' % (predicted_class_score * 100))
            ax.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        return fig

    def _convert_to_class_index(self, _class, dataset, default=None):
        if _class is None:
            return default
        else:
            if not isinstance(_class, int):
                return dataset.class_to_idx[_class]
            else:
                return _class


class SingleClassImageClassifierTrainer(Trainer, SingleClassImageClassifierMixin):
    """
    A trainer with some methods that are handy when you're training an image classifier model.  Specifically a model
    that classifies images into a single class per image.

    New in 0.10.0 (Used to be called ClassifierTrainer)
    """
    pass


class ClassifierTrainer(Trainer, SingleClassImageClassifierMixin):
    """
    Warning:
        This class has been deprecated for :py:class:`~sconce.trainers.SingleClassImageClassifierTrainer` and will be
        removed soon.  It will continue to work for now, but please update your code accordingly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('WARNING: ClassifierTrainer is deprecated as of 0.10.0, and will be removed soon.  Use '
            '"SingleClassImageClassifierTrainer" instead.')

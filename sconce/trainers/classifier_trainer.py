from abc import ABC
from scipy import sparse
from sconce.trainer import Trainer
from matplotlib import pyplot as plt

import seaborn as sn
import numpy as np


__all__ = ['ClassifierMixin', 'ClassifierTrainer']


class ClassifierMixin(ABC):
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
        matrix = self.get_confusion_matrix(data_generator=data_generator)

        defaults = {'cmap': 'YlGnBu', 'annot': True, 'fmt': 'd'}
        ax = sn.heatmap(matrix, **{**defaults, **heatmap_kwargs})

        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0)
        ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(),
                rotation=0, ha='right')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        return ax

    def plot_samples(self, predicted_label,
            true_label=None,
            data_generator=None,
            sort_by='rising predicted label score',
            num_samples=7,
            num_cols=7,
            figure_width=15,
            image_height=3,
            cache_results=True):

        if true_label is None:
            true_label = predicted_label

        if data_generator is None:
            data_generator = self.test_data_generator

        run_model_results = self._run_model_on_generator(data_generator,
                cache_results=cache_results)

        images = run_model_results['inputs']
        targets = run_model_results['targets']
        outputs = run_model_results['outputs']

        predicted_targets = np.argmax(outputs, axis=1)
        keep_idxs = ((targets == true_label) &
                     (predicted_targets == predicted_label))
        kept_images = images[keep_idxs]
        predicted_label_scores = np.exp(outputs[keep_idxs, predicted_label])
        true_label_scores = np.exp(outputs[keep_idxs, true_label])

        kept_images = np.array(kept_images)
        predicted_label_scores = np.array(predicted_label_scores)
        true_label_scores = np.array(true_label_scores)

        sort_fns = {
            'rising predicted label score': lambda p, t: np.argsort(p),
            'falling predicted label score': lambda p, t: np.argsort(p)[::-1],
            'rising true label score': lambda p, t: np.argsort(t),
            'falling true label score': lambda p, t: np.argsort(t)[::-1],
        }

        sort_fn = sort_fns[sort_by]
        sort_key = sort_fn(predicted_label_scores, true_label_scores)
        sorted_kept_images = kept_images[sort_key]
        sorted_predicted_label_scores = predicted_label_scores[sort_key]
        sorted_true_label_scores = true_label_scores[sort_key]

        if num_samples < len(kept_images):
            print(f'Showing only the first {num_samples} of '
                  f'{len(kept_images)} images')

        num_samples = min(num_samples, len(kept_images))
        num_rows = -(-num_samples // num_cols)
        fig = plt.figure(figsize=(figure_width, image_height * num_rows))

        for i in range(num_samples):
            image = sorted_kept_images[i]
            predicted_label_score = sorted_predicted_label_scores[i]
            true_label_score = sorted_true_label_scores[i]

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
            if true_label != predicted_label:
                ax.set_title('p: %2.1f%%\nt: %2.1f%%' % (
                    predicted_label_score * 100, true_label_score * 100))
            else:
                ax.set_title('%2.1f%%' % (predicted_label_score * 100))
            ax.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        return fig


class ClassifierTrainer(Trainer, ClassifierMixin):
    pass

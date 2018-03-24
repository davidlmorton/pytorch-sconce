from abc import ABC
from scipy import sparse
from sconce.trainer import Trainer

import seaborn as sn
import numpy as np


__all__ = ['ClassifierMixin', 'ClassifierTrainer']


class ClassifierMixin(ABC):
    def get_confusion_matrix(self, data_generator=None):
        if data_generator is None:
            data_generator = self.test_data_generator

        matrix = None
        data_generator.reset()
        for i in range(len(data_generator)):
            inputs, targets = data_generator.next()
            out_dict = self._run_model(inputs, targets, train=False)
            y_pred = np.argmax(out_dict['outputs'].cpu().data.numpy(), axis=1)
            y_true = out_dict['targets'].cpu().data.numpy()

            this_matrix = sparse.coo_matrix(
                    (np.ones(len(targets)), (y_pred, y_true)),
                    dtype='uint32').toarray()
            if matrix is None:
                matrix = this_matrix
            else:
                matrix += this_matrix
        return matrix

    def get_classification_accuracy(self, data_generator=None):
        if data_generator is None:
            data_generator = self.test_data_generator

        matrix = self.get_confusion_matrix(data_generator=data_generator)
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

    def plot_misclassified_samples(self, true_label):
        pass


class ClassifierTrainer(Trainer, ClassifierMixin):
    pass

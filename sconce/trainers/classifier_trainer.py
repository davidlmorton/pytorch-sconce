from abc import ABC
from sconce.trainer import Trainer

import numpy as np


__all__ = ['ClassifierMixin', 'ClassifierTrainer']


class ClassifierMixin(ABC):
    def get_classification_accuracy(self):
        num_correct = 0
        self.test_data_generator.reset()
        for i in range(len(self.test_data_generator)):
            inputs, targets = self.test_data_generator.next()
            out_dict = self._run_model(inputs, targets, train=False)
            y_out = np.argmax(out_dict['outputs'].cpu().data.numpy(), axis=1)
            y_in = out_dict['targets'].cpu().data.numpy()
            num_correct += (y_out - y_in == 0).sum()
        return num_correct / self.test_data_generator.num_samples

    def plot_confusion_matrix(self):
        pass

    def plot_misclassified_samples(self, true_label):
        pass


class ClassifierTrainer(Trainer, ClassifierMixin):
    pass

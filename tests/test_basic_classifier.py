# flake8: noqa
from sconce.data_generators import ImageDataGenerator
from sconce.schedules import Cosine
from sconce.trainers import ClassifierTrainer
from sconce.models import BasicClassifier
from torch import optim

import os
import torch
import unittest


class TestBasicClassifier(unittest.TestCase):
    def _test_file(self, *relative_path):
        base_path = os.path.dirname(__file__)
        return os.path.join(base_path, *relative_path)

    def test_full_run_from_yaml(self):

        filename = self._test_file('basic_classifier.yaml')
        model = BasicClassifier.new_from_yaml_filename(filename)

        training_generator = ImageDataGenerator.from_torchvision()
        test_generator = ImageDataGenerator.from_torchvision(train=False)

        if torch.cuda.is_available():
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        model.set_optimizer(optim.SGD, lr=1e-4, momentum=0.9, weight_decay=1e-4)

        trainer = ClassifierTrainer(model=model,
            training_data_generator=training_generator,
            test_data_generator=test_generator)

        self.assertLess(trainer.get_classification_accuracy(), 0.2)

        model.set_schedule('learning_rate', Cosine(initial_value=1e-1, final_value=3e-2))
        trainer.train(num_epochs=3)
        trainer.monitor.dataframe_monitor.plot(skip_first=30, smooth_window=5)

        self.assertGreater(trainer.get_classification_accuracy(), 0.95)

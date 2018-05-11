# flake8: noqa
from sconce.data_generators import SingleClassImageDataGenerator
from sconce.rate_controllers import CosineRateController, TriangleRateController
from sconce.trainers import SingleClassImageClassifierTrainer
from sconce.models import MultilayerPerceptron
from torch import optim

import os
import torch
import unittest


class TestMultilayerPerceptron(unittest.TestCase):
    def _test_file(self, *relative_path):
        base_path = os.path.dirname(__file__)
        return os.path.join(base_path, *relative_path)

    def test_full_run_from_yaml(self):
        filename = self._test_file('multilayer_perceptron.yaml')
        model = MultilayerPerceptron.new_from_yaml_filename(filename)

        training_generator = SingleClassImageDataGenerator.from_torchvision()
        test_generator = SingleClassImageDataGenerator.from_torchvision(train=False)

        if torch.cuda.is_available():
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        optimizer = optim.SGD(model.parameters(), lr=1e-4,
                momentum=0.9, weight_decay=1e-4)

        trainer = SingleClassImageClassifierTrainer(model=model, optimizer=optimizer,
            training_data_generator=training_generator,
            test_data_generator=test_generator)

        self.assertLess(trainer.get_classification_accuracy(), 0.2)

        rate_controller = CosineRateController(
                max_learning_rate=1e-1,
                min_learning_rate=3e-2)
        trainer.train(num_epochs=3, rate_controller=rate_controller)
        trainer.monitor.dataframe_monitor.plot(skip_first=30, smooth_window=5)

        acc = trainer.get_classification_accuracy()
        print(f"Accuracy: {acc}")
        self.assertGreater(acc, 0.90)

    def test_run_with_batch_multiplier(self):
        filename = self._test_file('multilayer_perceptron.yaml')
        model = MultilayerPerceptron.new_from_yaml_filename(filename)

        training_generator = SingleClassImageDataGenerator.from_torchvision(batch_size=100)
        test_generator = SingleClassImageDataGenerator.from_torchvision(batch_size=100,
                train=False)

        if torch.cuda.is_available():
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        optimizer = optim.SGD(model.parameters(), lr=1e-4,
                momentum=0.90, weight_decay=1e-4)

        trainer = SingleClassImageClassifierTrainer(model=model, optimizer=optimizer,
            training_data_generator=training_generator,
            test_data_generator=test_generator)

        survey_monitor = trainer.survey_learning_rate(min_learning_rate=1e-4,
                max_learning_rate=100)
        max_lr = survey_monitor.dataframe_monitor.df.learning_rate.max()
        print(f"Max learning rate tried: {max_lr}")
        self.assertLess(max_lr, 50)

        self.assertLess(trainer.get_classification_accuracy(), 0.2)

        rate_controller = TriangleRateController(
                max_learning_rate=5e-1,
                min_learning_rate=5e-2)
        trainer.train(num_epochs=3, rate_controller=rate_controller,
                batch_multiplier=5)

        acc = trainer.get_classification_accuracy()
        print(f"Accuracy: {acc}")
        self.assertGreater(acc, 0.80)

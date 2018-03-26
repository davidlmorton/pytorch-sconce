# flake8: noqa
from sconce.data_generator import DataGenerator
from sconce.rate_controllers import CosineRateController
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
        RANDOM_SEED = 1
        torch.manual_seed(RANDOM_SEED)

        filename = self._test_file('basic_classifier.yaml')
        model = BasicClassifier.new_from_yaml_filename(filename)

        training_generator = DataGenerator.from_pytorch()
        test_generator = DataGenerator.from_pytorch(train=False)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        optimizer = optim.SGD(model.parameters(), lr=1e-4,
                momentum=0.9, weight_decay=1e-4)

        trainer = ClassifierTrainer(model=model, optimizer=optimizer,
            training_data_generator=training_generator,
            test_data_generator=test_generator)

        self.assertTrue(trainer.get_classification_accuracy() < 0.2)

        rate_controller = CosineRateController(
                max_learning_rate=1e-1,
                min_learning_rate=3e-2)
        trainer.train(num_epochs=3, rate_controller=rate_controller)
        trainer.monitor.dataframe_monitor.plot(skip_first=30, smooth_window=5)

        self.assertTrue(trainer.get_classification_accuracy() > 0.95)

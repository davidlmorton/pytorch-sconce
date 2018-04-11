# flake8: noqa
from sconce.data_generator import DataGenerator
from sconce.rate_controllers import TriangleRateController
from sconce.trainers import ClassifierTrainer
from sconce.models import WideResnetImageClassifier
from torch import optim

import os
import torch
import unittest


class TestWideResnetImageClassifier(unittest.TestCase):
    def test_full_run(self):
        model = WideResnetImageClassifier(image_channels=1,
                depth=10, widening_factor=1)

        training_generator = DataGenerator.from_pytorch()
        test_generator = DataGenerator.from_pytorch(train=False)

        if torch.cuda.is_available():
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        optimizer = optim.SGD(model.parameters(), lr=1e-4,
                momentum=0.9, weight_decay=1e-4)

        trainer = ClassifierTrainer(model=model, optimizer=optimizer,
            training_data_generator=training_generator,
            test_data_generator=test_generator)

        self.assertTrue(trainer.get_classification_accuracy() < 0.2)

        rate_controller = TriangleRateController(
                max_learning_rate=1e-1,
                min_learning_rate=3e-2)
        trainer.train(num_epochs=1, rate_controller=rate_controller)

        acc = trainer.get_classification_accuracy()
        print(f"Accuracy: {acc}")
        self.assertTrue(acc > 0.90)

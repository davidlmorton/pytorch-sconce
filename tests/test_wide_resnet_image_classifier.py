# flake8: noqa
from sconce.data_generators import SingleClassImageDataGenerator
from sconce.rate_controllers import TriangleRateController
from sconce.trainers import SingleClassImageClassifierTrainer
from sconce.models import WideResnetImageClassifier
from torch import optim

import os
import torch
import unittest


class TestWideResnetImageClassifier(unittest.TestCase):
    def test_full_run(self):
        model = WideResnetImageClassifier(image_channels=1,
                depth=10, widening_factor=1)

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

        rate_controller = TriangleRateController(
                max_learning_rate=5e-2,
                min_learning_rate=5e-3)
        trainer.train(num_epochs=2, rate_controller=rate_controller)

        acc = trainer.get_classification_accuracy()
        print(f"Accuracy: {acc}")
        self.assertGreater(acc, 0.90)

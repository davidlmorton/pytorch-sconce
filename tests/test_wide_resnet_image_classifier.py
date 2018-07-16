# flake8: noqa
from sconce.data_feeds import SingleClassImageFeed
from sconce.schedules import Triangle
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

        training_feed = SingleClassImageFeed.from_torchvision()
        validation_feed = SingleClassImageFeed.from_torchvision(train=False)

        if torch.cuda.is_available():
            model.cuda()
            training_feed.cuda()
            validation_feed.cuda()

        model.set_optimizer(optim.SGD, lr=1e-4, momentum=0.9, weight_decay=1e-4)

        trainer = SingleClassImageClassifierTrainer(model=model,
            training_feed=training_feed,
            validation_feed=validation_feed)

        self.assertLess(trainer.get_classification_accuracy(), 0.2)

        num_steps = trainer.get_num_steps(num_epochs=2)
        model.set_schedule('learning_rate', Triangle(initial_value=5e-3, peak_value=5e-2))
        trainer.train(num_epochs=2)

        acc = trainer.get_classification_accuracy()
        print(f"Accuracy: {acc}")
        self.assertGreater(acc, 0.90)

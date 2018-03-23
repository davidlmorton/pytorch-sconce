# flake8: noqa
from sconce.data_generator import DataGenerator
from sconce.rate_controllers import CosineRateController
from sconce.trainers import ClassifierTrainer
from sconce.models import basic_classifier as bc
from torch import optim

import torch
import unittest


class TestBasicClassifier(unittest.TestCase):
    def test_full_run(self):
        RANDOM_SEED = 1
        torch.manual_seed(RANDOM_SEED)

        keys =      ('out_channels', 'stride', 'padding', 'kernel_size')
        values = ( # ==============  ========  =========  =============
                    (16,             1,        4,         9),
                    (8,              2,        1,         3),
                    (8,              2,        1,         3),
                    (8,              2,        1,         3),
                    (8,              2,        1,         3),
        )

        in_channels = 1
        convolutional_layers = []
        for values in values:
            kwargs = dict(zip(keys, values))
            convolutional_layers.append(bc.ConvolutionalLayer(
                in_channels=in_channels, **kwargs))
            in_channels = kwargs['out_channels']

        model = bc.BasicClassifier(image_width=28, image_height=28,
                    convolutional_layers=convolutional_layers,
                    dropouts=[0.4, 0.8],
                    fully_connected_sizes=[100, 100])

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

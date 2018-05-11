# pytorch-sconce
A library for training pytorch models

## Installation

```
pip install --process-dependency-links pytorch-sconce
```
Unfortunately the --process-dependency-links flag is needed until our contributions to `torchvision` have been merged
and we can depend on the version released on pypi instead of our fork.

## Documentation

You can find documentation [here](https://davidlmorton.github.io/pytorch-sconce).

## Pytorch-sconce flavored fast.ai notebooks

You can find Jupyter notebooks that follow the lessons from the fast.ai deep learning courses
[here](https://github.com/davidlmorton/fastai-course-sconce).

## Getting Started

We haven't had time to work on this documentation much yet.
Until then, you should look to the tests to see how the library works and what can be done with it.
Below are the contents of tests/test_basic_classifier.py:

```python
# flake8: noqa
from sconce.data_generators import ImageDataGenerator
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

        filename = self._test_file('basic_classifier.yaml')
        model = BasicClassifier.new_from_yaml_filename(filename)

        training_generator = ImageDataGenerator.from_torchvision()
        test_generator = ImageDataGenerator.from_torchvision(train=False)

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

        rate_controller = CosineRateController(
                max_learning_rate=1e-1,
                min_learning_rate=3e-2)
        trainer.train(num_epochs=3, rate_controller=rate_controller)
        trainer.monitor.dataframe_monitor.plot(skip_first=30, smooth_window=5)

        self.assertTrue(trainer.get_classification_accuracy() > 0.95)
```

With tests/basic_classifier.yaml looking like this:
```yaml
---
# Values for MNIST and FashionMNIST
image_height: 28
image_width: 28
image_channels: 1
num_categories: 10

# Remaining values are not related to the dataset
convolutional_layer_attributes: ["out_channels", "stride", "padding", "kernel_size"]
convolutional_layer_values:  [ # ==============  ========  =========  =============
                                [16,             1,        4,         9],
                                [8,              2,        1,         3],
                                [8,              2,        1,         3],
                                [8,              2,        1,         3],
                                [8,              2,        1,         3],
]

fully_connected_layer_attributes: ['out_size', 'dropout']
fully_connected_layer_values:  [ # ======      =========
                                  [100,        0.4],
                                  [100,        0.8],
]
```

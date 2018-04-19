from .layers import FullyConnectedLayer
from torch import nn
from torch.nn import functional as F

import numpy as np
import yaml


class MultilayerPerceptron(nn.Module):
    """
    A basic 2D image multi-layer perceptron built up of a number of densly connected layers.

    Loss:
        This model uses cross-entropy for the loss.

    Metrics:
        classification_accuracy: [0.0, 1.0] the fraction of correctly predicted labels.

    Arguments:
        image_height (int): image height in pixels.
        image_width (int): image width in pixels.
        image_channels (int): number of channels in the input images.
        layer_kwargs (list[dict]): a list of dictionaries describing layers. See
            :py:class:`~sconce.models.layers.fully_connected_layer.FullyConnectedLayer` for details.
        num_categories (int): [2, inf) the number of different image classes.
    """
    def __init__(self, image_height, image_width, image_channels,
            layer_kwargs,
            num_categories=10):
        super().__init__()

        in_channels = image_channels
        h = image_height
        w = image_width

        fc_layers = []
        num_channels = in_channels
        fc_size = w * h * num_channels
        for kwargs in layer_kwargs:
            layer = FullyConnectedLayer(in_size=fc_size,
                activation=nn.ReLU(), **kwargs)
            fc_layers.append(layer)
            fc_size = kwargs['out_size']
        self.non_final_layers = nn.ModuleList(fc_layers)

        self.final_layer = FullyConnectedLayer(fc_size,
                num_categories,
                with_batchnorm=False,
                activation=nn.LogSoftmax(dim=-1))

    @classmethod
    def new_from_yaml_filename(cls, yaml_filename):
        """
        Construct a new MultilayerPerceptron from a yaml file.

        Arguments:
            filename (path): the filename of a yaml file.  See
                :py:meth:`~sconce.models.basic_classifier.BasicClassifier.new_from_yaml_file`
                for more details.
        """
        with open(yaml_filename) as yaml_file:
            return cls.new_from_yaml_file(yaml_file)

    @classmethod
    def new_from_yaml_file(cls, yaml_file):
        """
        Construct a new MultilayerPerceptron from a yaml file.

        Arguments:
            yaml_file (file): a file-like object that yaml contents can be read from.

        Example yaml file contents:

        .. code-block:: yaml

            ---
            # Values for MNIST and FashionMNIST
            image_height: 28
            image_width: 28
            image_channels: 1
            num_categories: 10

            layer_attributes: ['out_size', 'dropout', 'with_batchnorm']
            layer_values:  [ # ======      =========  ================
                              [100,        0.4,       true],
                              [100,        0.8,       true],
            ]
        """
        yaml_data = yaml.load(yaml_file)

        layer_kwargs = []
        keys = yaml_data.pop('layer_attributes')
        for values in yaml_data.pop('layer_values'):
            kwargs = dict(zip(keys, values))
            layer_kwargs.append(kwargs)

        return cls(layer_kwargs=layer_kwargs, **yaml_data)

    def forward(self, inputs, **kwargs):
        x = inputs.view(inputs.size()[0], -1)
        for layer in self.non_final_layers:
            x = layer(x)

        outputs = self.final_layer(x)
        return {'outputs': outputs}

    def calculate_loss(self, targets, outputs, **kwargs):
        return {'loss': F.nll_loss(input=outputs, target=targets)}

    def calculate_metrics(self, targets, outputs, **kwargs):
        y_out = np.argmax(outputs.cpu().data.numpy(), axis=1)
        y_in = targets.cpu().data.numpy()
        num_correct = (y_out - y_in == 0).sum()
        classification_accuracy = num_correct / len(y_in)
        return {'classification_accuracy': classification_accuracy}

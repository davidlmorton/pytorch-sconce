from .layers import FullyConnectedLayer, Convolution2dLayer
from torch import nn
from torch.nn import functional as F

import numpy as np
import yaml


class BasicClassifier(nn.Module):
    """
    A basic 2D image classifier built up of some number of convolutional layers followed by some number of densly
    connected layers.

    Loss:
        This model uses cross-entropy for the loss.

    Metrics:
        classification_accuracy: [0.0, 1.0] the fraction of correctly predicted labels.

    Arguments:
        image_height (int): image height in pixels.
        image_width (int): image width in pixels.
        image_channels (int): number of channels in the input images.
        convolutional_layer_kwargs (list[dict]): a list of dictionaries describing the convolutional layers. See
            :py:class:`~sconce.models.layers.convolution2d_layer.Convolution2dLayer` for details.
        fully_connected_layer_kwargs (list[dict]): a list of dictionaries describing the fully connected layers. See
            :py:class:`~sconce.models.layers.fully_connected_layer.FullyConnectedLayer` for details.
        num_categories (int): [2, inf) the number of different image classes.
    """
    def __init__(self, image_height, image_width, image_channels,
            convolutional_layer_kwargs,
            fully_connected_layer_kwargs,
            num_categories=10):
        super().__init__()

        in_channels = image_channels
        h = image_height
        w = image_width
        convolutional_layers = []
        for kwargs in convolutional_layer_kwargs:
            layer = Convolution2dLayer(in_channels=in_channels, **kwargs)
            convolutional_layers.append(layer)
            in_channels = kwargs['out_channels']
            h = layer.out_height(h)
            w = layer.out_width(w)

        self.convolutional_layers = nn.ModuleList(convolutional_layers)

        fc_layers = []
        num_channels = in_channels
        fc_size = w * h * num_channels
        for kwargs in fully_connected_layer_kwargs:
            layer = FullyConnectedLayer(in_size=fc_size,
                activation=nn.ReLU(), **kwargs)
            fc_layers.append(layer)
            fc_size = kwargs['out_size']
        self.fully_connected_layers = nn.ModuleList(fc_layers)

        self.final_layer = FullyConnectedLayer(fc_size,
                num_categories,
                with_batchnorm=False,
                activation=nn.LogSoftmax(dim=-1))

    @property
    def layers(self):
        return ([x for x in self.fully_connected_layers] +
                [x for x in self.convolutional_layers] + [self.final_layer])

    def freeze_batchnorm_layers(self):
        for layer in self.layers:
            layer.freeze_batchnorm()

    def unfreeze_batchnorm_layers(self):
        for layer in self.layers:
            layer.unfreeze_batchnorm()

    @classmethod
    def new_from_yaml_filename(cls, yaml_filename):
        """
        Construct a new BasicClassifier from a yaml file.

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
        Construct a new BasicClassifier from a yaml file.

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
        """
        yaml_data = yaml.load(yaml_file)

        convolutional_layer_kwargs = []
        keys = yaml_data.pop('convolutional_layer_attributes')
        for values in yaml_data.pop('convolutional_layer_values'):
            kwargs = dict(zip(keys, values))
            convolutional_layer_kwargs.append(kwargs)

        fully_connected_layer_kwargs = []
        keys = yaml_data.pop('fully_connected_layer_attributes')
        for values in yaml_data.pop('fully_connected_layer_values'):
            kwargs = dict(zip(keys, values))
            fully_connected_layer_kwargs.append(kwargs)

        return cls(convolutional_layer_kwargs=convolutional_layer_kwargs,
                fully_connected_layer_kwargs=fully_connected_layer_kwargs,
                **yaml_data)

    def forward(self, inputs, **kwargs):
        x = inputs
        for i, layer in enumerate(self.convolutional_layers):
            x = layer(x)

        x = x.view(inputs.size()[0], -1)
        for layer in self.fully_connected_layers:
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

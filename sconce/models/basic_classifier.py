from .layers import FullyConnectedLayer, Convolution2dLayer
from torch import nn
from torch.nn import functional as F

import numpy as np
import yaml


class BasicClassifier(nn.Module):
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
        with open(yaml_filename) as yaml_file:
            return cls.new_from_yaml_file(yaml_file)

    @classmethod
    def new_from_yaml_file(cls, yaml_file):
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

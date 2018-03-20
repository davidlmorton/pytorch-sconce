from torch import nn
from torch.nn import functional as F

import numpy as np


class ConvolutionalLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1):
        super().__init__()

        def make_tuple(i):
            if isinstance(i, int):
                return (i, i)
            else:
                return i

        self.stride = make_tuple(stride)
        self.kernel_size = make_tuple(kernel_size)
        self.padding = make_tuple(padding)
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding=self.padding)
        self.relu = nn.ReLU()

    def out_height(self, in_height):
        numerator = (in_height + 2 * self.padding[0] -
                (self.kernel_size[0] - 1) - 1)
        return (numerator // self.stride[0]) + 1

    def out_width(self, in_width):
        numerator = (in_width + 2 * self.padding[1] -
                (self.kernel_size[1] - 1) - 1)
        return (numerator // self.stride[1]) + 1

    def forward(self, x_in):
        x = self.bn(x_in)
        x = self.conv(x)
        x = self.relu(x)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size, activation, dropout=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_size)
        self.fc = nn.Linear(in_size, out_size)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in):
        x = self.bn(x_in)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class BasicClassifier(nn.Module):
    def __init__(self, image_height, image_width, convolutional_layers,
            fully_connected_sizes=[100, 200],
            dropouts=[0.30, 0.60],
            num_categories=10):
        super().__init__()

        self.convolutional_layers = nn.ModuleList(convolutional_layers)

        h = image_height
        w = image_width
        for layer in convolutional_layers:
            h = layer.out_height(h)
            w = layer.out_width(w)
            num_channels = layer.out_channels

        fc_size = fully_connected_sizes[0]
        fc_layers = [FullyConnectedLayer(in_size=w * h * num_channels,
            out_size=fc_size, dropout=dropouts[0], activation=nn.ReLU())]
        for size, dropout in zip(fully_connected_sizes[1:], dropouts[1:]):
            fc_layers.append(FullyConnectedLayer(in_size=fc_size,
                out_size=size, dropout=dropout, activation=nn.ReLU()))
            fc_size = size
        self.fully_connected_layers = nn.ModuleList(fc_layers)

        self.final_layer = FullyConnectedLayer(fc_size,
                num_categories, nn.LogSoftmax(dim=-1))

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

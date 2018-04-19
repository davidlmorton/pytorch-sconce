from .layers import Convolution2dLayer
from torch import nn
from torch.nn import functional as F

import numpy as np


class AdaptiveAveragePooling2dLayer(nn.Module):
    def __init__(self, in_channels, output_size,
            inplace_activation=False,
            preactivate=False,
            with_batchnorm=True):
        super().__init__()
        self.preactivate = preactivate

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.relu = nn.ReLU(inplace=inplace_activation)
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, x):
        if self.with_batchnorm:
            x = self.bn(x)

        if self.preactivate:
            x = self.relu(x)
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = self.relu(x)

        return x


class WideResnetBlock_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = Convolution2dLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                inplace_activation=True,
                preactivate=True)

        self.conv2 = Convolution2dLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                inplace_activation=True,
                preactivate=True)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.skip_conv = Convolution2dLayer(
                    kernel_size=1,
                    padding=0,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    inplace_activation=True,
                    preactivate=True)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.conv2(x)
        if self.in_channels == self.out_channels:
            x = x + x_in
        else:
            x = x + self.skip_conv(x_in)
        return x


class WideResnetGroup_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = WideResnetBlock_3x3(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride)
            else:
                block = WideResnetBlock_3x3(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class WideResnetImageClassifier(nn.Module):
    """
    A wide resnet image classifier, based on `this paper`_

    .. _this paper: http://arxiv.org/abs/1605.07146

    Loss:
        This model uses cross-entropy for the loss.

    Metrics:
        classification_accuracy: [0.0, 1.0] the fraction of correctly predicted labels.

    Arguments:
        image_channels (int): number of channels in the input images.
        depth (int): total number of convolutional layers in the network.  This
            should be divisible by (6n + 4) where n is a positive integer.
        widening_factor (int): [1, inf) determines how many convolutional
            channels are in the network (see paper above for details).
        num_categories (int): [2, inf) the number of different image classes.
    """
    def __init__(self, image_channels=1,
            depth=28,
            widening_factor=10,
            num_categories=10):
        super().__init__()

        assert (depth - 4) % 6 == 0, '"depth" parameter should be 6n + 4'
        n = (depth - 4) // 6

        widths = [width * widening_factor for width in (16, 32, 64)]

        self.conv1 = Convolution2dLayer(
                in_channels=image_channels,
                out_channels=16,
                stride=1,
                inplace_activation=True,
                preactivate=True)

        self.group1 = WideResnetGroup_3x3(in_channels=16,
                out_channels=widths[0],
                stride=1,
                num_blocks=n)

        self.group2 = WideResnetGroup_3x3(in_channels=widths[0],
                out_channels=widths[1],
                stride=2,
                num_blocks=n)

        self.group3 = WideResnetGroup_3x3(in_channels=widths[1],
                out_channels=widths[2],
                stride=2,
                num_blocks=n)

        self.conv2 = Convolution2dLayer(
                kernel_size=1,
                padding=0,
                in_channels=widths[2],
                out_channels=num_categories,
                stride=1,
                inplace_activation=True,
                preactivate=True)

        self.pool = AdaptiveAveragePooling2dLayer(
                in_channels=num_categories,
                output_size=1,
                inplace_activation=True,
                preactivate=True)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = inputs
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(inputs.size()[0], -1)
        x = self.log_softmax(x)
        return {'outputs': x}

    def calculate_loss(self, targets, outputs, **kwargs):
        return {'loss': F.nll_loss(input=outputs, target=targets)}

    def calculate_metrics(self, targets, outputs, **kwargs):
        y_out = np.argmax(outputs.cpu().data.numpy(), axis=1)
        y_in = targets.cpu().data.numpy()
        num_correct = (y_out - y_in == 0).sum()
        classification_accuracy = num_correct / len(y_in)
        return {'classification_accuracy': classification_accuracy}

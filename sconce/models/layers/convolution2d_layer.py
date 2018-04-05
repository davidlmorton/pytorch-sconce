from torch import nn


class Convolution2dLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1,
            with_batchnorm=True):
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

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
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
        if self.with_batchnorm:
            x = self.bn(x_in)
        else:
            x = x_in
        x = self.conv(x)
        x = self.relu(x)
        return x

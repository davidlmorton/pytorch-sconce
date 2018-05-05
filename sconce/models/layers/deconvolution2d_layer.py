from torch import nn


class Deconvolution2dLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1,
            output_padding=1,
            inplace_activation=False,
            preactivate=False,
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
        self.output_padding = make_tuple(output_padding)
        self.out_channels = out_channels
        self.preactivate = preactivate

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels,
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding=self.padding,
                output_padding=output_padding)
        self.relu = nn.ReLU(inplace=inplace_activation)

    def out_height(self, in_height):
        return (in_height - 1) * self.stride[0] - 2 * self.padding[0] +\
               kernel_size[0] + output_padding[0]

    def out_width(self, in_width):
        return (in_width - 1) * self.stride[1] - 2 * self.padding[1] +\
               kernel_size[1] + output_padding[1]

    def forward(self, x):
        if self.with_batchnorm:
            x = self.bn(x)

        if self.preactivate:
            x = self.relu(x)
            x = self.deconv(x)
        else:
            x = self.deconv(x)
            x = self.relu(x)

        return x

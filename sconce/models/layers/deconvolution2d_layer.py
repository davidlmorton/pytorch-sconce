from torch import nn


def make_tuple(i):
    if isinstance(i, int):
        return (i, i)
    else:
        return i


class Deconvolution2dLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1,
            output_padding=1,
            inplace_activation=False,
            preactivate=False,
            with_batchnorm=True):
        super().__init__()

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

    @classmethod
    def matching_input_and_output_size(cls, input_size, output_size, **kwargs):
        candidate = cls(**{**kwargs, 'output_padding': 0})
        h, w = candidate.out_size(input_size)
        output_padding = [output_size[0] - h, output_size[1] - w]
        padding = list(make_tuple(kwargs.get('padding', 0)))

        if output_padding[0] < 0:
            padding[0] += 1
            output_padding[0] += 2

        if output_padding[1] < 0:
            padding[1] += 1
            output_padding[1] += 2

        kwargs['padding'] = padding
        kwargs['output_padding'] = output_padding
        return cls(**kwargs)

    def out_height(self, in_height):
        return (in_height - 1) * self.stride[0] - 2 * self.padding[0] +\
               self.kernel_size[0] + self.output_padding[0]

    def out_width(self, in_width):
        return (in_width - 1) * self.stride[1] - 2 * self.padding[1] +\
               self.kernel_size[1] + self.output_padding[1]

    def out_size(self, in_size):
        return (self.out_height(in_size[0]), self.out_width(in_size[1]))

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

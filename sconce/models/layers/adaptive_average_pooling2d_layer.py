from torch import nn


class AdaptiveAveragePooling2dLayer(nn.Module):
    def __init__(self, in_channels, output_size,
            activation=nn.ReLU(),
            preactivate=False,
            with_batchnorm=True):
        super().__init__()
        self.preactivate = preactivate

        if activation is None:
            activation = lambda x: x
        self.activation = activation

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, x):
        if self.with_batchnorm:
            x = self.bn(x)

        if self.preactivate:
            x = self.activation(x)
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = self.activation(x)

        return x

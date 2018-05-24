from torch import nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size,
            activation=nn.ReLU(),
            with_batchnorm=True,
            dropout=0.0):
        super().__init__()

        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn = nn.BatchNorm1d(in_size)

        self.fc = nn.Linear(in_size, out_size)

        self._dropout_value = dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, x_in):
        if self.with_batchnorm:
            x = self.bn(x_in)
        else:
            x = x_in

        x = self.fc(x)

        if self._dropout_value > 0.0:
            x = self.dropout(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

from torch import nn
from torch.nn import functional as F
from sconce.models.layers import Convolution2dLayer, Deconvolution2dLayer


class BasicConvolutionalAutoencoder(nn.Module):
    """
    A basic 2D image autoencoder built up of convolutional layers, three each in the encoder and the decoder.

    Loss:
        This model uses binary cross-entropy for the loss.

    Metrics:
        None

    Arguments:
        image_channels (int): the number of channels in the input images.
        conv_channels (list of int): a list of length three of integers describing the number of channels in each of the
            three convolutional layers.
    """
    def __init__(self, image_channels, conv_channels):
        super().__init__()
        self.conv1 = Convolution2dLayer(
                in_channels=image_channels,
                out_channels=conv_channels[0])

        self.conv2 = Convolution2dLayer(
                in_channels=conv_channels[0],
                out_channels=conv_channels[1])

        self.conv3 = Convolution2dLayer(
                in_channels=conv_channels[1],
                out_channels=conv_channels[2],
                padding=2)

        self.deconv1 = Deconvolution2dLayer(
                in_channels=conv_channels[2],
                out_channels=conv_channels[1],
                padding=2,
                output_padding=0)

        self.deconv2 = Deconvolution2dLayer(
                in_channels=conv_channels[1],
                out_channels=conv_channels[0])

        self.deconv3 = Deconvolution2dLayer(
                in_channels=conv_channels[0],
                out_channels=image_channels,
                preactivate=True)

    def encode(self, x_in, **kwargs):
        x = self.conv1(x_in)
        x = self.conv2(x)
        x_latent = self.conv3(x)
        return x_latent

    def decode(self, x_latent):
        x = self.deconv1(x_latent)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x_out = nn.Sigmoid()(x)
        return x_out

    def forward(self, inputs, **kwargs):
        x_latent = self.encode(inputs)
        x_out = self.decode(x_latent)
        return {'outputs': x_out}

    def calculate_loss(self, inputs, outputs, **kwargs):
        reconstruction_loss = F.binary_cross_entropy(outputs,
                inputs.view_as(outputs))
        return {'loss': reconstruction_loss}

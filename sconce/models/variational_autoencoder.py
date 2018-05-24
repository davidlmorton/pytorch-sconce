from torch import nn
from torch.nn import functional as F
from sconce.models.layers import Convolution2dLayer, Deconvolution2dLayer, FullyConnectedLayer

import torch


class VariationalAutoencoder(nn.Module):
    """
    A variational autoencoder built up of convolutional layers and dense layers in the encoder and decoder.

    Loss:
        This model uses binary cross entropy for the reconstruction loss and KL Divergence for the latent
        representation loss.

    Metrics:
        None

    Arguments:
        image_channels (int): the number of channels in the input images.
        conv_channels (list of int): a list (of length three) of integers describing the number of channels in each
            of the three convolutional layers.
        hidden_sizes (list of int): a list (of length two) of integers describing the number of hidden units in
            each hidden layer.
        latent_size (int): the size of the latent representation.
    """
    def __init__(self, image_size, image_channels,
            conv_channels=[32, 32, 32],
            hidden_sizes=[256, 256],
            latent_size=10,
            beta=1.0):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.conv_channels = conv_channels
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.beta = 1.0

        self.conv1 = Convolution2dLayer(
                padding=0,
                kernel_size=2,
                in_channels=image_channels,
                out_channels=conv_channels[0])
        size1 = self.conv1.out_size(image_size)

        self.conv2 = Convolution2dLayer(
                padding=0,
                kernel_size=3,
                in_channels=conv_channels[0],
                out_channels=conv_channels[1])
        size2 = self.conv2.out_size(size1)

        self.conv3 = Convolution2dLayer(
                padding=0,
                kernel_size=3,
                in_channels=conv_channels[1],
                out_channels=conv_channels[2])
        size3 = self.conv3.out_size(size2)
        self.size3 = size3
        self.num_conv_activations = size3[0] * size3[1] * conv_channels[2]

        self.encoder_hidden1 = FullyConnectedLayer(
                in_size=self.num_conv_activations,
                out_size=hidden_sizes[0])
        self.encoder_hidden2 = FullyConnectedLayer(
                in_size=hidden_sizes[0],
                out_size=hidden_sizes[1])

        self.mean = FullyConnectedLayer(
                in_size=hidden_sizes[1],
                out_size=latent_size,
                activation=None)
        self.log_variance = FullyConnectedLayer(
                in_size=hidden_sizes[1],
                out_size=latent_size,
                activation=None)

        self.decoder_hidden_latent = FullyConnectedLayer(
                in_size=latent_size,
                out_size=hidden_sizes[1])
        self.decoder_hidden2 = FullyConnectedLayer(
                in_size=hidden_sizes[1],
                out_size=hidden_sizes[0])
        self.decoder_hidden1 = FullyConnectedLayer(
                in_size=hidden_sizes[0],
                out_size=self.num_conv_activations)

        self.deconv3 = Deconvolution2dLayer.matching_input_and_output_size(
                input_size=size3,
                output_size=size2,
                in_channels=conv_channels[2],
                out_channels=conv_channels[1],
                kernel_size=3,
                padding=0)

        self.deconv2 = Deconvolution2dLayer.matching_input_and_output_size(
                input_size=size2,
                output_size=size1,
                in_channels=conv_channels[1],
                out_channels=conv_channels[0],
                padding=0,
                kernel_size=3)

        # preactivate so we can tack on the sigmoid activation afterwards.
        self.deconv1 = Deconvolution2dLayer.matching_input_and_output_size(
                input_size=size1,
                output_size=self.image_size,
                in_channels=conv_channels[0],
                out_channels=image_channels,
                kernel_size=2,
                padding=0,
                preactivate=True)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x_in, **kwargs):
        x = self.conv1(x_in)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_conv_activations)
        x = self.encoder_hidden1(x)
        x = self.encoder_hidden2(x)
        mu = self.mean(x)
        logvar = self.log_variance(x)
        return mu, logvar

    def decode(self, x_latent):
        x = self.decoder_hidden_latent(x_latent)
        x = self.decoder_hidden2(x)
        x = self.decoder_hidden1(x)
        x = x.view(-1, self.conv_channels[2], self.size3[0], self.size3[1])
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        x_out = nn.Sigmoid()(x)
        return x_out

    def forward(self, inputs, **kwargs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        x_out = self.decode(z)
        return {'outputs': x_out, 'mu': mu, 'logvar': logvar}

    def calculate_loss(self, inputs, outputs, mu, logvar, **kwargs):
        reconstruction_loss = F.binary_cross_entropy(outputs,
                inputs.view_as(outputs))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        latent_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {'loss': reconstruction_loss + self.beta * latent_loss,
                'latent_loss': latent_loss,
                'reconstruction_loss': reconstruction_loss}

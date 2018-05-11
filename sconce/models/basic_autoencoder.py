from torch import nn
from torch.nn import functional as F
from sconce.models.layers import FullyConnectedLayer


class BasicAutoencoder(nn.Module):
    """
    A basic 2D image autoencoder built up of fully connected layers, three each in the encoder and the decoder.

    Loss:
        This model uses binary cross-entropy for the loss.

    Metrics:
        None

    Arguments:
        image_height (int): image height in pixels.
        image_width (int): image width in pixels.
        hidden_size (int): the number of activations in each of the 4 hidden layers.
        latent_size (int): the number of activations in the latent representation (encoder output).
    """
    def __init__(self, image_height, image_width, hidden_size, latent_size):
        super().__init__()
        self.num_pixels = image_height * image_width

        self.fc1 = FullyConnectedLayer(in_size=self.num_pixels,
                out_size=hidden_size,
                activation=nn.ReLU())

        self.fc2 = FullyConnectedLayer(in_size=hidden_size,
                out_size=hidden_size,
                activation=nn.ReLU())

        self.fc3 = FullyConnectedLayer(in_size=hidden_size,
                out_size=latent_size,
                activation=nn.ReLU())

        self.fc4 = FullyConnectedLayer(in_size=latent_size,
                out_size=hidden_size,
                activation=nn.ReLU())

        self.fc5 = FullyConnectedLayer(in_size=hidden_size,
                out_size=hidden_size,
                activation=nn.ReLU())

        self.fc6 = FullyConnectedLayer(in_size=hidden_size,
                out_size=self.num_pixels,
                activation=nn.Sigmoid())

    def encode(self, inputs, **kwargs):
        encoder_input = inputs.view(-1, self.num_pixels)
        x = self.fc1(encoder_input)
        x = self.fc2(x)
        x_latent = self.fc3(x)
        return x_latent

    def decode(self, x_latent):
        x = self.fc4(x_latent)
        x = self.fc5(x)
        outputs = self.fc6(x)
        return outputs

    def forward(self, inputs, **kwargs):
        x_latent = self.encode(inputs)
        outputs = self.decode(x_latent)
        return {'outputs': outputs}

    def calculate_loss(self, inputs, outputs, **kwargs):
        reconstruction_loss = F.binary_cross_entropy(outputs,
                inputs.view_as(outputs))
        return {'loss': reconstruction_loss}

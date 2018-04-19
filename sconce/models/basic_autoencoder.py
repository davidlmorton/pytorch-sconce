from torch import nn
from torch.nn import functional as F


class BasicAutoencoder(nn.Module):
    """
    A basic 2D image autoencoder built up of densly connected layers, two each in the encoder and the decoder.

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

        self.bn1 = nn.BatchNorm1d(self.num_pixels)
        self.fc1 = nn.Linear(self.num_pixels, hidden_size)

        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)

        self.bn3 = nn.BatchNorm1d(latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)

        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.num_pixels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, inputs, **kwargs):
        encoder_input = inputs.view(-1, self.num_pixels)
        x = self.bn1(encoder_input)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.bn2(x)
        x = self.fc2(x)
        x_latent = self.relu(x)
        return x_latent

    def decode(self, x_latent):
        x = self.bn3(x_latent)
        x = self.fc3(x)
        x = self.relu(x)

        x = self.bn4(x)
        x = self.fc4(x)
        outputs = self.sigmoid(x)
        return outputs

    def forward(self, inputs, **kwargs):
        x_latent = self.encode(inputs)
        outputs = self.decode(x_latent)
        return {'outputs': outputs}

    def calculate_loss(self, inputs, outputs, **kwargs):
        reconstruction_loss = F.binary_cross_entropy(outputs,
                inputs.view_as(outputs))
        return {'loss': reconstruction_loss}

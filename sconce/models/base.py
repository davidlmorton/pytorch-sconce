from abc import ABC, abstractmethod
from torch import nn

import numpy as np


class Model(ABC, nn.Module):
    """
    The base class of all Models in Sconce.  It is only an interface, describing what must be implemented
    if you want to define a model.
    """
    def __init__(self):
        super(ABC, self).__init__()
        super(nn.Module, self).__init__()

    @abstractmethod
    def forward(self, *, inputs, targets, **kwargs):
        """
        It must accept arbitrary keyword arguments.  The base class of trainer will pass
        `inputs` and `targets`, but subclasses may modify that behavior to include other keyword arguments.

        It must return a dictionary. The dictionary is expected to include at least the key `outputs`
        but may include any other keys you like.  The value of the key `outputs` is expected to be
        the :py:class:`torch.Tensor` output of the model, used for calculating the loss.
        """

    @abstractmethod
    def calculate_loss(self, *, inputs, outputs, targets, **kwargs):
        """
        This method must accept arbitrary keyword arguments.  The base class of trainer will pass `inputs`,
        `outputs`, and `targets`, but subclasses may modify that behavior to include other keyword arguments.

        It must return a dictionary.  The dictionary is expected to include at least the key 'loss', but may
        include any otehr keys you like.  The value of the key `loss` is expected to be the :py:class:`torch.Tensor`
        output of the loss function, used to back-propagate the gradients used by the optimizer.
        """

    def calculate_metrics(self, *, inputs, outputs, targets, loss, **kwargs):
        """
        This method must accept arbitrary keyword arguments.  The base class of trainer will pass `inputs`,
        `outputs`, `targets`, and `loss`, but subclasses may modify that behavior to include other keyword arguments.

        It must return a dictionary.  No restrictions are made on the keys or values of this dictionary.
        """
        return {}

    def get_trainable_parameters(self):
        """
        The trainable parameters that the models has.
        """
        return list(filter(lambda p: p.requires_grad, self.parameters()))

    def get_num_trainable_parameters(self):
        """
        The number of trainable parameters that the models has.
        """
        return sum([np.prod(p.size()) for p in self.get_trainable_parameters()])

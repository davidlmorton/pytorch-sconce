# flake8: noqa
"""
A model in Sconce is just a :py:class:`torch.nn.Module` with a few restrictions
(see note below).  Sconce comes with a few example models to get you started.

Note:
    For a model to be used by a :py:class:`~sconce.trainer.Trainer`, the model must satisfy a few constraints.

    #. It must accept arbitrary keyword arguments in it's ``forward`` method.  The base class of trainer will pass
       `inputs` and `targets`, but subclasses may modify that behavior to include other keyword arguments.
    #. It must return a dictionary from it's ``forward`` method.  The dictionary is expected to include at least
       the key `outputs` but may include any other keys you like.  The value of the key `outputs` is expected to be
       the :py:class:`torch.Tensor` output of the model, used for calculating the loss.
    #. It must define a ``calculate_loss`` method.  This method must accept arbitrary keyword arguments.  The base
       class of trainer will pass `inputs`, `outputs`, and `targets`, but subclasses may modify that behavior to
       include other keyword arguments.
    #. It must return a dictionary form it's ``calculate_loss`` method.  The dictionary is expected to include at
       least the key 'loss', but may include any otehr keys you like.  The value of the key `loss` is expected to
       be the :py:class:`torch.Tensor` output of the loss function, used to back-propagate the
       gradients used by the optimizer.
    #. It may define a ``calculate_metrics`` method.  This method must accept arbitrary keyword arguments.  The
       base class of trainer will pass `inputs`, `outputs`, `targets`, and `loss`, but subclasses may modify that
       behavior to include other keyword arguments.
    #. If it defines a ``calculate_metrics`` method, it must return a dictionary.  No restrictions are made on the
       keys or values of this dictionary.
"""
from .basic_autoencoder import BasicAutoencoder
from .basic_classifier import BasicClassifier
from .basic_convolutional_autoencoder import BasicConvolutionalAutoencoder
from .multilayer_perceptron import MultilayerPerceptron
from .variational_autoencoder import VariationalAutoencoder
from .wide_resnet_image_classifier import WideResnetImageClassifier

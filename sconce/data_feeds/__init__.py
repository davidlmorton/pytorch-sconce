# flake8: noqa
from .base import DataFeed
from .image import ImageFeed
from .multi_class_image import MultiClassImageFeed
from .single_class_image import SingleClassImageFeed

def print_deprecation_warning(argument_prefix=''):
    print(f"WARNING: The {argument_prefix}data_generator argument is deprecated as of 1.2.0, and will "
          f"be removed soon.  Please use {argument_prefix}feed instead.")

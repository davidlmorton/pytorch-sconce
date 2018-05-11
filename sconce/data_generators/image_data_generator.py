from sconce.data_generators import SingleClassImageDataGenerator


class ImageDataGenerator(SingleClassImageDataGenerator):
    """
    Warning:
        This class has been deprecated for :py:class:`~sconce.data_generators.SingleClassImageDataGenerator` and will be
        removed soon.  It will continue to work for now, but please update your code accordingly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('WARNING: ImageDataGenerator is deprecated as of 0.10.0, and will be removed soon.  Use '
            '"SingleClassImageDataGenerator" instead.')

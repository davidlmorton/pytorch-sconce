import numpy as np

__all__ = ['NHot']


class NHot(object):
    """
    Converts a list of indices to a n-hot encoded vector.

    arguments:
        size (int): the size of the returned array

    example:
        >>> l = [3, 7, 2, 1]
        >>> t = NHot(size=10)
        >>> t(l)
        array([0., 1., 1., 1., 0., 0., 0., 1., 0., 0.])
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, indices):
        """
        arguments:
            indices (iterable of integers): the indices of the classes that an image belongs to.

        returns:
            array: the n-hot encoded representation of the classes that an image belongs to.
        """
        indices_array = np.array(indices)
        target = np.zeros(self.size, dtype=np.float32)
        target[indices_array] = 1
        return target

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'

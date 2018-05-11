from torch.utils import data


class DataGenerator:
    """
    A thin wrapper around a :py:class:`~torch.utils.data.DataLoader` that
    automatically yields tuples of :py:class:`torch.Tensor` (that
    live on cpu or on cuda).
    A DataGenerator will iterate endlessly.

    Like the underlying :py:class:`~torch.utils.data.DataLoader`, a
    DataGenerator's ``__next__`` method yields two values, which we refer to as
    the `inputs` and the `targets`.

    Arguments:
        data_loader (:py:class:`~torch.utils.data.DataLoader`): the wrapped
            data_loader.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._inputs_cuda = False
        self._targets_cuda = False
        self.reset()

    def cuda(self, device=None):
        """
        Put the `inputs` and `targets` (yielded by this DataGenerator) on the
        specified device.

        Arguments:
            device (int or bool or dict): if int or bool, sets the behavior for
                both `inputs` and `targets`. To set them individually, pass a
                dictionary with keys {'inputs', 'targets'} instead. See
                :py:meth:`torch.Tensor.cuda` for details.

        Example:
            >>> g = DataGenerator.from_dataset(dataset, batch_size=100)
            >>> g.cuda()
            >>> g.next()
            (Tensor containing:
             [torch.cuda.FloatTensor of size 100x1x28x28 (GPU 0)],
             Tensor containing:
             [torch.cuda.LongTensor of size 100 (GPU 0)])
            >>> g.cuda(False)
            >>> g.next()
            (Tensor containing:
             [torch.FloatTensor of size 100x1x28x28],
             Tensor containing:
             [torch.LongTensor of size 100])
            >>> g.cuda(device={'inputs':0, 'targets':1})
            >>> g.next()
            (Tensor containing:
             [torch.cuda.FloatTensor of size 100x1x28x28 (GPU 0)],
             Tensor containing:
             [torch.cuda.LongTensor of size 100 (GPU 1)])
        """
        if isinstance(device, dict):
            for key, value in device.items():
                if key == 'inputs':
                    self._inputs_cuda = value
                elif key == 'targets':
                    self._targets_cuda = value
                else:
                    raise RuntimeError(f"Invalid key for 'device' argument: "
                            f"({key}) expected to be in ('inputs', 'targets').")
        else:
            self._inputs_cuda = device
            self._targets_cuda = device

    def reset(self):
        """
        Start iterating through the data_loader from the begining.
        """
        self._iterator = iter(self.data_loader)

    @property
    def dataset(self):
        """
        the wrapped data_loader's :py:class:`~torch.utils.data.Dataset`
        """
        return self.data_loader.dataset

    @property
    def real_dataset(self):
        """
        the wrapped data_loader's :py:class:`~torch.utils.data.Dataset` reaching through any Subsets
        """
        return self._real_dataset(self.dataset)

    def _real_dataset(self, dataset):
        result = dataset
        if isinstance(result, data.dataset.Subset):
            return self._real_dataset(result.dataset)
        else:
            return result

    @property
    def batch_size(self):
        """
        the wrapped data_loader's batch_size
        """
        return self.data_loader.batch_size

    @property
    def num_samples(self):
        """
        the ``len`` of the wrapped data_loader's
        :py:class:`~torch.utils.data.Dataset`
        """
        return len(self.dataset)

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def preprocess(self, inputs, targets):
        if self._inputs_cuda is False:
            inputs = inputs.cpu()
        else:
            inputs = inputs.cuda(self._inputs_cuda)

        if self._targets_cuda is False:
            targets = targets.cpu()
        else:
            targets = targets.cuda(self._targets_cuda)

        return inputs, targets

    def next(self):
        try:
            inputs, targets = self._iterator.next()
        except StopIteration:
            self.reset()
            inputs, targets = self._iterator.next()
        return self.preprocess(inputs, targets)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """
        Create a DataGenerator from an instantiated dataset.

        Arguments:
            dataset (:py:class:`~torch.utils.data.Dataset`): the pytorch
                dataset.
            **kwargs: passed directly to the
                :py:class:`~torch.utils.data.DataLoader`) constructor.
        """
        data_loader = data.DataLoader(dataset, **kwargs)
        return cls(data_loader)

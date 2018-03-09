from torch.autograd import Variable


class DataGenerator:
    """
    A thin wrapper around a <data_loader> that turns torch.Tensors into
    torch.Variables (that live on cpu or on cuda) and iterates endlessly.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._inputs_cuda = False
        self._targets_cuda = False
        self.reset()

    def cuda(self, device=None):
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
        self._iterator = iter(self.data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    @property
    def batch_size(self):
        return self.data_loader.batch_size

    @property
    def num_samples(self):
        return len(self.dataset)

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def preprocess(self, inputs, targets):
        inputs = Variable(inputs)
        targets = Variable(targets)

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

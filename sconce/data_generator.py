class DataGenerator:
    """Thin wrapper around a <data_loader> that iterates endlessly"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.reset()

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

    def next(self):
        try:
            return self._iterator.next()
        except StopIteration:
            self.reset()
            return self._iterator.next()

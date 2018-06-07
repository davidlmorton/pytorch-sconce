from torch.utils import data


class Subset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        super().__init__()
        self.__dict__['dataset'] = dataset
        self.__dict__['indices'] = indices
        if hasattr(dataset, 'targets'):
            self.__dict__['targets'] = [dataset.targets[idx]
                    for idx in indices]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __setattr__(self, name, value):
        return setattr(self.dataset, name, value)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

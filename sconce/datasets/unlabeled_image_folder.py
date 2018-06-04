from torchvision.datasets import folder

import os
import os.path
import torch.utils.data as data


class UnlabeledImageFolder(data.Dataset):
    """
    A Dataset that reads images from a folder.  Targets are the paths to the images.

    Arguments:
        root (string): directory where the images can be found.
        loader (callable, optional): a function to load a sample given its path.
        extensions (list[string], optinoal): a list of allowed extensions. E.g, ``['.jpg', '.tif']``
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root,
            loader=folder.default_loader,
            extensions=folder.IMG_EXTENSIONS,
            transform=None,
            target_transform=None):

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

        self.paths = self.get_paths(self.root)

    def get_paths(self, root):
        paths = []
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            paths.append(path)
        return paths

    def get_sample(self, index):
        path = self.paths[index]
        return self.loader(path)

    def get_target(self, index):
        return self.paths[index]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where sample is the image, and target the path to the image
        """
        sample = self.get_sample(index)
        target = self.get_target(index)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

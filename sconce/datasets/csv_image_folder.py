from PIL import Image
from torchvision.datasets import folder
from collections import defaultdict

import csv
import glob
import os
import os.path
import torch.utils.data as data


class CsvImageFolder(data.Dataset):
    """
    A Dataset that reads images from a folder and labels from a csv file.

    Arguments:
        root (string): directory where the images can be found.
        csv_path (string): the path to the csv file containing image filenames and labels.
        filename_key (string, optional): the column header of the csv for the column that contains image filenames
            (without extensions).
        labels_key (string, optional): the column header of the csv for the column that contains labels for each image.
        csv_delimiter (string, optional): the character(s) used to separate fields in the csv file.
        loader (callable, optional): a function to load a sample given its path.
        extensions (list[string], optinoal): a list of allowed extensions. E.g, ``['.jpg', '.tif']``
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, csv_path,
            filename_key='image_name',
            labels_key='tags',
            csv_delimiter=',',
            labels_delimiter=' ',
            loader=folder.default_loader,
            extensions=folder.IMG_EXTENSIONS,
            transform=None,
            target_transform=None):

        self.root = root
        self.csv_path = csv_path
        self.filename_key = filename_key
        self.labels_key = labels_key
        self.csv_delimiter = csv_delimiter
        self.labels_delimiter = labels_delimiter
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

        self._found_extensions = None
        self._load_found_extensions()

        self._load()

    def _load(self):
        self._rows = []
        all_labels = set()
        with open(self.csv_path) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=self.csv_delimiter)
            for row in reader:
                self._rows.append(row)
                labels = self._get_labels(row)
                all_labels.update(labels)

        self._labels = sorted(all_labels)
        self._label_idxs = {label:i for i, label in enumerate(self._labels)}

    def _get_path(self, base_filename):
        found_extensions = self.found_extensions
        if base_filename in found_extensions:
            found_extensions = found_extensions[base_filename]
            for extension in self.extensions:
                if extension in found_extensions:
                    return os.path.join(self.root, '%s%s' % (base_filename, extension))
        raise RuntimeError(f"No image file with base filename ({base_filename}) "
            f"found in folder ({self.root}), valid extensions are: {self.extensions}")

    @property
    def found_extensions(self):
        if self._found_extensions is None:
            self._found_extensions = self._load_found_extensions()
        return self._found_extensions

    def _load_found_extensions(self):
        found_extensions = defaultdict(list)
        for filename in os.listdir(self.root):
            base, ext = os.path.splitext(filename)
            found_extensions[base].append(ext)
        return dict(found_extensions)

    def _get_labels(self, row):
        return set(row[self.labels_key].split(self.labels_delimiter))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where sample is the image, and target is an array of indices of the target
                class.
        """
        row = self._rows[index]

        base_filename = row[self.filename_key]
        path = self._get_path(base_filename)
        sample = self.loader(path)

        labels = self._get_labels(row)
        target = [self._label_idxs[label] for label in labels]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self._rows)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Number of tags: {}\n'.format(len(self._labels))
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

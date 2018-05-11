from collections import defaultdict
from sconce import transforms
from torchvision.datasets import folder

import csv
import os
import os.path
import torch.utils.data as data


class CsvImageFolder(data.Dataset):
    """
    A Dataset that reads images from a folder and classes from a csv file.

    Arguments:
        root (string): directory where the images can be found.
        csv_path (string): the path to the csv file containing image filenames and classes.
        filename_key (string, optional): the column header of the csv for the column that contains image filenames
            (without extensions).
        classes_key (string, optional): the column header of the csv for the column that contains classes for each
            image.
        csv_delimiter (string, optional): the character(s) used to separate fields in the csv file.
        loader (callable, optional): a function to load a sample given its path.
        extensions (list[string], optinoal): a list of allowed extensions. E.g, ``['.jpg', '.tif']``
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    Attributes:
        class_to_idx (dict): a dictionary mapping class names to indices.
        classes (list[string]): the human readable names of the classes that images can belong to.
        paths (list[string]): for each image, the path to the image on disk.
        targets (list[list[int]]): for each image, a list of class indices to which that image belongs.
    """

    def __init__(self, root, csv_path,
            filename_key='image_name',
            classes_key='tags',
            csv_delimiter=',',
            classes_delimiter=' ',
            loader=folder.default_loader,
            extensions=folder.IMG_EXTENSIONS,
            transform=None,
            target_transform=transforms.NHot):

        self.root = root
        self.csv_path = csv_path
        self.filename_key = filename_key
        self.classes_key = classes_key
        self.csv_delimiter = csv_delimiter
        self.classes_delimiter = classes_delimiter
        self.loader = loader
        self.extensions = extensions
        self.transform = transform

        self._found_extensions = None
        self._load_found_extensions()

        self.class_to_idx = {}
        self.classes = []
        self.paths = []
        self.targets = []
        self._load()

        if target_transform is transforms.NHot:
            self.target_transform = transforms.NHot(size=len(self.classes))
        else:
            self.target_transform = target_transform

    def _load_found_extensions(self):
        found_extensions = defaultdict(list)
        for filename in os.listdir(self.root):
            base, ext = os.path.splitext(filename)
            found_extensions[base].append(ext)
        return dict(found_extensions)

    @property
    def found_extensions(self):
        if self._found_extensions is None:
            self._found_extensions = self._load_found_extensions()
        return self._found_extensions

    def _load(self):
        classes_set = set()
        classes_list = []
        with open(self.csv_path) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=self.csv_delimiter)
            for row in reader:
                filename = self._get_filename(row)
                path = self._get_path(filename)
                self.paths.append(path)

                classes = self._get_classes(row)
                classes_list.append(classes)
                classes_set.update(classes)

        self.classes = sorted(classes_set)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        targets = []
        for classes in classes_list:
            targets.append([self.class_to_idx[_class] for _class in classes])
        self.targets = targets

    def _get_filename(self, row):
        return row[self.filename_key]

    def _get_path(self, base_filename):
        found_extensions = self.found_extensions
        if base_filename in found_extensions:
            found_extensions = found_extensions[base_filename]
            for extension in self.extensions:
                if extension in found_extensions:
                    return os.path.join(self.root, '%s%s' % (base_filename, extension))
        raise RuntimeError(f"No image file with base filename ({base_filename}) "
            f"found in folder ({self.root}), valid extensions are: {self.extensions}")

    def _get_classes(self, row):
        return row[self.classes_key].split(self.classes_delimiter)

    def get_sample(self, index):
        path = self.paths[index]
        return self.loader(path)

    def get_target(self, index):
        return self.targets[index]

    def _get_target(self, row):
        classes = self._get_classes(row)
        return [self.class_to_idx[_class] for _class in classes]

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where sample is the image, and target is an array of indices of the target
                class.
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
        fmt_str += '    Number of different classes: {}\n'.format(len(self.classes))
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

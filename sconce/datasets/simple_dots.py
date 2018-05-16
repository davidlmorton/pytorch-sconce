from PIL import Image, ImageDraw
from torchvision import transforms

import numpy as np
import torch
import torch.utils.data as data


def generate_circle_image(x, y, image_size=(40, 40), circle_radius=4):
    assert x >= 0, 'x must be between 0.0 and 1.0'
    assert x <= 1, 'x must be between 0.0 and 1.0'
    assert y >= 0, 'y must be between 0.0 and 1.0'
    assert y <= 1, 'y must be between 0.0 and 1.0'

    x_pixel = (image_size[0] - 2 * circle_radius - 1) * x + circle_radius
    y_pixel = (image_size[1] - 2 * circle_radius - 1) * y + circle_radius

    image = Image.new('L', image_size)
    draw = ImageDraw.Draw(image)
    upper_left = (x_pixel - circle_radius, y_pixel - circle_radius)
    lower_right = (x_pixel + circle_radius, y_pixel + circle_radius)
    draw.ellipse((*upper_left, *lower_right), fill='white', outline='white')

    return image


class SimpleDots(data.Dataset):
    """
    A Dataset where each image is a white dot on a black background, the targets are the coordinates of the dot.

    Arguments:
        image_size (tuple of int): the size of the image (in pixels).
        num_images (int): the number of images to generate for this dataset.  Best if the sqrt of this number is an
            integer.
        circle_radius (float): the size of the dot (in pixels).

    New in 0.11.0
    """
    def __init__(self, circle_radius=4, image_size=(30, 30), num_images=400):
        self.circle_radius = circle_radius
        self.image_size = image_size
        self.num_images = num_images

        self.samples, self.targets = self._generate()

    def _generate(self):
        num_xs = int(np.sqrt(self.num_images))
        num_ys = self.num_images // num_xs
        xs = np.linspace(0.0, 1.0, num_xs)
        ys = np.linspace(0.0, 1.0, num_ys)

        transform = transforms.ToTensor()

        images = []
        coords = []
        for x in xs:
            for y in ys:
                pil_image = generate_circle_image(x, y,
                        image_size=self.image_size, circle_radius=self.circle_radius)
                image = transform(pil_image)
                images.append(image)
                coords.append(torch.Tensor((x, y)))

        samples = torch.stack(images)
        targets = torch.stack(coords)

        return samples, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where sample is the image, and target is the x, y coordinates of the dot in the
                image.
        """
        return self.samples[index], self.targets[index]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'SimpleDots Dataset\n'
        fmt_str += f'    Size of images: {self.image_size}\n'
        fmt_str += f'    Radius of circle: {self.circle_radius}\n'
        fmt_str += f'    Number of images: {len(self)}\n'
        return fmt_str

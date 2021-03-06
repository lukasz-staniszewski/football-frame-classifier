import torch
from os import listdir
from os.path import isfile, join
import torchvision
import os


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.files_names = [
            f
            for f in listdir(self.images_folder)
            if isfile(join(self.images_folder, f))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, index):
        filename = self.files_names[index]
        image = torchvision.io.read_image(
            os.path.join(self.images_folder, filename)
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, filename


class OneImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, image_name, transform=None):
        self.image_folder = image_folder
        self.image_name = image_name
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        filename = self.image_name
        image = torchvision.io.read_image(
            os.path.join(self.image_folder, filename)
        )
        if self.transform is not None:
            image = self.transform(image)
        return image
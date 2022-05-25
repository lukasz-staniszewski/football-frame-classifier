import torch
import pandas as pd
import torchvision
import os


class FramesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, class2index, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = class2index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df["basename"][index]
        label = self.class2index[self.df["category"][index]]
        image = torchvision.io.read_image(
            os.path.join(self.images_folder, filename)
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label

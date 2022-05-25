import torch
import pandas as pd
import torchvision
import os


class FramesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {
            "side_view": 0,
            "closeup": 1,
            "non_match": 2,
            "front_view": 3,
            "side_gate_view": 4,
            "aerial_view": 5,
            "wide_view": 6,
        }

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

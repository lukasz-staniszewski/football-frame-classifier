from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import FramesDataset


class FramesDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_folder,
        batch_size,
        csv_path,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
    ):
        trsfm = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize(size=(360, 640)),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        self.images_folder = images_folder
        self.dataset = FramesDataset(
            csv_path=csv_path,
            images_folder=self.images_folder,
            transform=trsfm,
        )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )

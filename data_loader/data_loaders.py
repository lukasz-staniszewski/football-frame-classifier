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
                transforms.Normalize(
                    (0.3683047, 0.42932022, 0.29250222),
                    (0.15938677, 0.16319054, 0.17476037),
                ),
            ]
        )
        self.class2index = {
            "side_view": 0,
            "closeup": 1,
            "non_match": 2,
            "front_view": 3,
            "side_gate_view": 4,
            "aerial_view": 5,
            "wide_view": 6,
        }
        self.images_folder = images_folder
        self.dataset = FramesDataset(
            csv_path=csv_path,
            images_folder=self.images_folder,
            class2index=self.class2index,
            transform=trsfm,
        )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )

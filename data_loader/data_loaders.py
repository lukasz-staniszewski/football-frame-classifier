from torchvision import datasets, transforms
from torchvision.transforms.transforms import RandomHorizontalFlip
from base import BaseDataLoader
from data_loader import FramesDataset, TestDataset
from torch.utils.data import ConcatDataset


class FramesDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_folder,
        batch_size,
        csv_path,
        csv_path_tf,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        is_with_aug=False,
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
        trsfm_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize(size=(360, 640)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(25),
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
        self.dataset_orig = FramesDataset(
            csv_path=csv_path,
            images_folder=self.images_folder,
            class2index=self.class2index,
            transform=trsfm,
        )
        if is_with_aug:
            self.dataset_aug = FramesDataset(
                csv_path=csv_path_tf,
                images_folder=self.images_folder,
                class2index=self.class2index,
                transform=trsfm_aug,
            )
            self.dataset = ConcatDataset([self.dataset_orig, self.dataset_aug])
        else:
            self.dataset = self.dataset_orig

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
        )


class TestDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_folder,
        batch_size,
        shuffle=False,
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
        self.index2class = {
            0: "side_view",
            1: "closeup",
            2: "non_match",
            3: "front_view",
            4: "side_gate_view",
            5: "aerial_view",
            6: "wide_view",
        }
        self.images_folder = images_folder
        self.dataset = TestDataset(images_folder=self.images_folder, transform=trsfm)

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
        )

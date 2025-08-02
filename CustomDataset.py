import torch
from PIL import Image
from typing import Dict, Tuple, List, Callable, Union
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from ColorDistorter import ColorDistorter
from CentralCropResize import CentralCropResize
from dataset_builder.core.utility import load_manifest_parquet


class CustomDataset(Dataset):
    """
    A custom dataset creator which loads a list of dictionaries contain image path and its label and convert it to tensor to used in training and evaluation

    Args:

        data (List[Dict[str, int]): Either a path to a parquet file containing image paths and labels or a list of tuples with image paths and labels.
        train (bool): If True, applies data augmentation. Defaults to True.
        img_size (Tuple[int, int]): The size to which images will be resized. Defaults to (160, 160).
    """

    def __init__(
        self,
        data: List[Dict[str, int]],
        train: bool = True,
        img_size: Tuple[int, int] = (160, 160),
    ):
        super().__init__()
        self.image_label_with_correct_labels = data
        self.image_paths = [item["image"] for item in data]
        self.labels = [item["label"] for item in data]
        self.train = train
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_label_with_correct_labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        # supress pyright warning
        image = Image.open(image_path).convert("RGB")  # type: ignore

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        color_ordering = worker_id % 4

        if self.train:
            transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.img_size, scale=(0.05, 1.0), ratio=(0.75, 1.33)
                    ),
                    transforms.RandomHorizontalFlip(),
                    ColorDistorter(ordering=color_ordering),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    CentralCropResize(central_fraction=0.875, size=self.img_size),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        image = transform(image)  # type: ignore

        return image, label  # type: ignore

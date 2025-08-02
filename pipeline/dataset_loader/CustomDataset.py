import torch
from PIL import Image
from typing import Tuple, List, Callable, Union
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from pipeline.preprocessing import ColorDistorter, CentralCropResize
from dataset_builder.core.utility import load_manifest_parquet


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images and labels from either a manifest file or a preloaded list.

    This dataset supports optional training transformations such as random cropping, flipping, and color distortion, and applies a simple central crop during validation.

    Args:
        data (Union[str, List[Tuple[str, int]]]):
            Either the path to a Parquet manifest file or a pre-loaded list of
            (image_path, label) tuples.
        train (bool, optional):
            Whether to apply training augmentations (True) or validation transformations (False).
            Defaults to True.
        img_size (Tuple[int, int], optional):
            Target size (width, height) to which images are resized. Defaults to (224, 224).

    Notes:
        - Training transformations include random resized crop, horizontal flip, and a custom color distortion whose variant depends on the worker ID (for multi-worker diversity).
        - Validation transformation applies a central crop and resize.
        - Images are loaded as RGB regardless of the original format.
        - Normalization during training maps image pixel values into [-1, 1].
    """

    def __init__(
        self,
        data: Union[str, List[Tuple[str, int]]],
        train: bool = True,
        img_size: Tuple[int, int] = (160, 160),
    ):
        if isinstance(data, str):
            self.image_label_with_correct_labels = load_manifest_parquet(data)
        else:
            self.image_label_with_correct_labels = data
        self.train = train
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_label_with_correct_labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.image_label_with_correct_labels[index]
        image = Image.open(img_path).convert("RGB")

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
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    CentralCropResize(central_fraction=0.875, size=self.img_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        image = transform(image)  # type: ignore

        return image, label  # type: ignore

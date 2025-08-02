from torch.utils.data import DataLoader
from .create_dataset import DatasetCreator
from CustomDataset import CustomDataset
from utility.utilities import manifest_generator_wrapper
from typing import Tuple, Dict, Any


class DataLoaderCreator:
    """
    Class to create a DataLoader for a given dataset.
    Args:
        batch_size (int): Size of each batch.
        num_workers (int): Number of worker threads for data loading.
        dominant_threshold (float): Threshold for dominant species.
        start_rank (int): Starting rank for dataset partitioning.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        dominant_threshold: float = 0.2,
        start_rank: int = 0,
        is_binary: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dominant_threshold = dominant_threshold
        self.start_rank = start_rank
        self.is_binary = is_binary

    def create_dataloader(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any], int, Dict[str, Any]]:
        """
        Create a DataLoader for the dataset.
        Returns:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation datat.
        """
        _, _, _, species_labels, _ = manifest_generator_wrapper(
            self.dominant_threshold, export=True
        )

        num_classes = len(species_labels.keys())
        print(f"Number of classes in the dataset: {num_classes}")
        if self.is_binary:
            num_classes = 2
        print(f"Number of species from manifest: {num_classes}")
        datacreator = DatasetCreator(number_of_dominant_classes=num_classes)

        # The actual number of classes in the dataset is NUM_SPECIES + 1 (including "Other" class)

        print(f"Species labels: {species_labels.keys()}")

        _, train, val, weights, label_map = datacreator.create_dataset(self.start_rank)
        train_dataset = CustomDataset(train, train=True, img_size=(160, 160))
        val_dataset = CustomDataset(val, train=False, img_size=(160, 160))

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        # dump species label to file:
        return train_loader, val_loader, weights, num_classes, label_map  # type: ignore

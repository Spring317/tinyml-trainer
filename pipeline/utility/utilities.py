import json
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset_builder import run_manifest_generator
from dataset_builder.core import load_config, validate_config
from dataset_builder.core.exceptions import ConfigError
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from pipeline.dataset_loader import CustomDataset


def get_device(use_cpu=False) -> torch.device:
    """
    Determines the computation device (CPU or GPU) to be used for training or inference.

    Args:
        use_cpu (bool, optional):
            If True, forces the use of CPU even if a CUDA-compatible GPU is available.
            Defaults to False.

    Returns:
        torch.device:
            A torch.device object representing either "cpu" or "cuda".
    """
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"GPU model: {torch.cuda.get_device_name(device)}")
        else:
            print("No GPU found")
    print(f"Using device: {device}")
    return device


def mobile_net_v3_large_builder(
    device: torch.device,
    num_outputs: Optional[int] = None,
    start_with_weight=False,
    path: Optional[str] = None,
):
    """
    Builds or loads a MobileNetV3-Large model, optionally customized for a specific number of output classes.

    This function supports two modes:
        - Load a full pre-trained model checkpoint from a specified path.
        - Build a new MobileNetV3-Large model from scratch (optionally using ImageNet weights), and modify its final classification layer to match the desired number of outputs.

    Args:
        device (torch.device):
            The device (CPU or GPU) on which the model will be loaded or created.
        num_outputs (Optional[int], optional):
            Number of output classes for the final classification layer.
            Required if building a new model. Ignored if loading from a checkpoint.
        start_with_weight (bool, optional):
            If True, initializes the model with default ImageNet pre-trained weights.
            Defaults to False (random initialization).
        path (Optional[str], optional):
            Path to a `.pth` file containing a serialized full model to load.
            If provided, `num_outputs` is ignored.

    Returns:
        torch.nn.Module:
            A MobileNetV3-Large model instance moved to the specified device.

    Notes:
        - When building a new model, the final classifier layer (`classifier[3]`) is replaced with a new `nn.Linear` to match the desired number of classes.
        - When loading from a checkpoint (`path` is given), the model is assumed to have the correct output dimension already.
        - Raises an assertion error if `num_outputs` is not provided when building a new model from scratch.
    """

    if path and not num_outputs:
        # load full model
        model = torch.load(path, map_location=device, weights_only=False)
        model = model.to(device)

    else:
        if start_with_weight:
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
        else:
            model = models.mobilenet_v3_large(weights=None)
        old_linear_layer = model.classifier[3]
        assert isinstance(old_linear_layer, nn.Linear), "Expected a Linear layer"
        assert isinstance(num_outputs, int), (
            "Expected an int for classification layer output"
        )
        model.classifier[3] = nn.Linear(old_linear_layer.in_features, num_outputs)
        model = model.to(device)

    return model


def convnext_large_builder(
    device: torch.device,
    num_outputs: Optional[int] = None,
    start_with_weight=False,
    path: Optional[str] = None,
    input_size: tuple = (160, 160),
):
    """
    Builds or loads a ConvNeXt-Large model, optionally customized for a specific number of output classes and input size.

    This function supports two modes:
        - Load a full pre-trained model checkpoint from a specified path.
        - Build a new ConvNeXt-Large model from scratch (optionally using ImageNet weights), and modify its final classification layer to match the desired number of outputs.

    Args:
        device (torch.device):
            The device (CPU or GPU) on which the model will be loaded or created.
        num_outputs (Optional[int], optional):
            Number of output classes for the final classification layer.
            Required if building a new model. Ignored if loading from a checkpoint.
        start_with_weight (bool, optional):
            If True, initializes the model with default ImageNet pre-trained weights.
            Defaults to False (random initialization).
        path (Optional[str], optional):
            Path to a `.pth` file containing a serialized full model to load.
            If provided, `num_outputs` is ignored.
        input_size (int, optional):
            Input image size (assumes square images). Defaults to 160.

    Returns:
        torch.nn.Module:
            A ConvNeXt-Large model instance moved to the specified device.

    Notes:
        - When building a new model, the final classifier layer (`classifier[2]`) is replaced with a new `nn.Linear` to match the desired number of classes.
        - When loading from a checkpoint (`path` is given), the model is assumed to have the correct output dimension already.
        - Raises an assertion error if `num_outputs` is not provided when building a new model from scratch.
        - For input sizes other than 224, the model adapts automatically through adaptive pooling.
    """

    if path:
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Checkpoint contains state_dict and metadata (from torch.save with dict)
                model = models.convnext_large(weights=None)
                num_classes = checkpoint["model_state_dict"][
                    "classifier.2.weight"
                ].shape[0]
                model.classifier[2] = nn.Linear(
                    model.classifier[2].in_features, num_classes
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model = model.to(device)
                print(f"Loaded model with {num_classes} output classes from checkpoint")

            elif isinstance(checkpoint, dict) and "classifier.2.weight" in checkpoint:
                # Checkpoint is a state_dict directly (from model.state_dict())
                model = models.convnext_large(weights=None)
                num_classes = checkpoint["classifier.2.weight"].shape[0]
                model.classifier[2] = nn.Linear(
                    model.classifier[2].in_features, num_classes
                )
                model.load_state_dict(checkpoint)
                model = model.to(device)
                print(f"Loaded model with {num_classes} output classes from state_dict")

            else:
                # Checkpoint is a full model (from torch.save(model, path))
                model = checkpoint.to(device)
                print("Loaded full model from checkpoint")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("The checkpoint file might be corrupted or in an unexpected format.")

            # Don't create a new model - this would give wrong results!
            # Instead, raise an error to alert the user
            raise RuntimeError(
                f"Failed to load checkpoint from '{path}'. "
                f"Please check if the file exists and is a valid PyTorch checkpoint. "
                f"Creating a new untrained model would give incorrect results."
            ) from e

    else:
        # Build new model
        if start_with_weight:
            model = models.convnext_large(
                weights=models.convnext.ConvNeXt_Large_Weights.IMAGENET1K_V1
            )
        else:
            model = models.convnext_large(weights=None)

        old_linear_layer = model.classifier[2]
        assert isinstance(old_linear_layer, nn.Linear), "Expected a Linear layer"
        assert isinstance(num_outputs, int), (
            "Expected an int for classification layer output"
        )
        model.classifier[2] = nn.Linear(old_linear_layer.in_features, num_outputs)

        # Handle 160x160 input size
        if input_size != 224:
            # ConvNeXt uses adaptive average pooling, so it should handle different input sizes
            # The model will automatically adapt to the input size through its adaptive pooling layer
            pass

        model = model.to(device)
        print(f"Model configured for {input_size}x{input_size} input size")

    return model


def dataloader_wrapper(
    train_dataset: CustomDataset,
    val_dataset: CustomDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """
    Creates training and validation DataLoaders from the given datasets with common configuration options.

    Args:
        train_dataset (CustomDataset):
            The dataset used for training.
        val_dataset (CustomDataset):
            The dataset used for validation.
        batch_size (int):
            Number of samples per batch to load.
        num_workers (int):
            Number of subprocesses to use for data loading.
        shuffle (bool, optional):
            Whether to shuffle the dataset at every epoch. Defaults to True.
        pin_memory (bool, optional):
            If True, the data loader will copy tensors into CUDA pinned memory before returning them. Defaults to True.
        persistent_workers (bool, optional):
            Whether to keep data loading workers alive between epochs.
            Improves efficiency when `num_workers > 0`. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader]:
            A tuple containing:
                - train_loader: DataLoader for the training set.
                - val_loader: DataLoader for the validation set.

    Notes:
        - Both training and validation loaders share the same batch size and worker configuration.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def train_one_epoch_amp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
):
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)

            # Tensor guard
            if labels.min() < 0 or labels.max() >= outputs.shape[1]:
                print("Invalid labels detected!")
                print(f"Labels: {labels}")
                print(f"Label min: {labels.min().item()} | max: {labels.max().item()}")
                print(f"Number of classes (output.shape[1]): {outputs.shape[1]}")
                raise ValueError("Label out of range for CrossEntropyLoss.")

            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.item():.3f}")
    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def get_support_list(
    species_composition: Union[str, Dict[str, int]], species_name: List[str]
) -> List[int]:
    """
    Retrieves a list of sample counts (support values) for each species name in a specified order.

    This function takes either a path to a JSON file or a dictionary containing species name-to-count mappings, and returns a list of support values aligned with the given list of species names.

    Args:
        species_composition (Union[str, Dict[str, int]]):
            Either a path to a JSON file or a dictionary mapping species names (str) to image counts (int).
        species_name (List[str]):
            List of species names for which support values should be extracted.
            The order of this list defines the order of the returned support values.

    Returns:
        List[int]:
            A list of support counts (image counts) corresponding to the input `species_name` list.
            If a species name is not found in the composition, 0 is returned for that species.
    """
    if isinstance(species_composition, str):
        with open(species_composition, "r") as f:
            species_count_dict = json.load(f)
    else:
        species_count_dict = species_composition
    total_support_list = [species_count_dict.get(name, 0) for name in species_name]
    return total_support_list


def generate_report(
    all_labels: List[int],
    all_preds: List[int],
    species_names: List[str],
    total_support_list: List[int],
    accuracy: float,
) -> pd.DataFrame:
    """
    Generates a detailed classification report as a pandas DataFrame, including per-class metrics
    and additional support statistics for validation and training data.

    Args:
        all_labels (List[int]):
            Ground-truth class labels for all validation samples.
        all_preds (List[int]):
            Predicted class labels for all validation samples.
        species_names (List[str]):
            Ordered list of species names corresponding to class indices.
        total_support_list (List[int]):
            Total number of available samples per species (i.e., train + val).
        accuracy (float):
            Overall classification accuracy to include in the report.
    """
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=species_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # extract support and compute training support
    val_support_series = report_df.loc[species_names, "support"].astype(int)
    total_support_series = pd.Series(total_support_list, index=species_names)
    train_support_series = total_support_list - val_support_series

    # add new column
    report_df.loc[species_names, "support"] = val_support_series
    report_df.loc[species_names, "train_support"] = train_support_series.astype(int)
    report_df.loc[species_names, "total_support"] = total_support_series.astype(int)

    # reorganize the report
    species_df = report_df.loc[species_names].copy()  # type: ignore
    summary_df = report_df.drop(index=species_names).copy()

    # sort species by f1-score (ascending)
    species_df = species_df.sort_values(by="f1-score", ascending=True)

    # combine details + summary
    report_df = pd.concat([species_df, summary_df])

    report_df.loc["accuracy"] = {  # type: ignore
        "precision": accuracy,
        "recall": 0,
        "f1-score": 0,
        "support": np.nan,
        "train_support": np.nan,
        "total_support": np.nan,
    }

    # clean up summary support rows
    for row in ["macro avg", "weighted avg"]:
        for col in ["support", "train_support", "total_support"]:
            if col in report_df.columns:
                report_df.loc[row, col] = np.nan
    return report_df


def manifest_generator_wrapper(
    dominant_threshold: Optional[float] = None, export: bool = False
) -> Tuple[
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    Dict[int, str],
    Dict[str, int],
]:
    """
    Wrapper function that loads configuration and calls `run_manifest_generator()` to generate train/val manifests and species composition based on a dominant class threshold.

    This function parses configuration from a YAML file (`./config.yaml`), applies optional overrides, validates the config, and calls the underlying manifest generation logic.

    Args:
        dominant_threshold (Optional[float], optional):
            Overrides the dominant species threshold from the config file if provided.
            If `None`, the threshold from `config["train_val_split"]["dominant_threshold"]` is used.
        export (bool, optional):
            Whether to export the full manifest, train/val splits, and composition JSONs.
            Defaults to False.

    Returns:
        Tuple[
            List[Tuple[str, int]],
            List[Tuple[str, int]],
            List[Tuple[str, int]],
            Dict[int, str],
            Dict[str, int]
        ]

        A tuple containing:
            - The complete list of images and their labels.
            - The training split of the dataset.
            - The validation split of the dataset.
            - A dictionary mapping label to the species name.
            - A dictionary mapping species names to their image count.
    """
    try:
        config = load_config("./config.yaml")
        # validate_config(config)
    except ConfigError as e:
        print(e)
        exit()

    target_classes = config["global"]["included_classes"]
    dst_dataset_path = config["paths"]["dst_dataset"]
    dst_dataset_name = os.path.basename(dst_dataset_path)
    output_path = config["paths"]["output_dir"]
    dst_properties_path = os.path.join(
        output_path, f"{dst_dataset_name}_composition.json"
    )
    train_size = config["train_val_split"]["train_size"]
    randomness = config["train_val_split"]["random_state"]
    if not dominant_threshold:
        dominant_threshold = config["train_val_split"]["dominant_threshold"]
    assert isinstance(dominant_threshold, float), "Invalid dominant threshold."

    return run_manifest_generator(
        dst_dataset_path,
        dst_dataset_path,
        dst_properties_path,
        train_size,
        randomness,
        target_classes,
        dominant_threshold,
        export=export,
    )


def calculate_weight_cross_entropy(
    species_composition: Union[str, Dict[str, int]],
    species_labels: Union[str, Dict[int, str]],
):
    """
    Computes class weights for imbalanced classification using inverse frequency, suitable for use with `torch.nn.CrossEntropyLoss(weight=...)`.

    Accepts either file paths or preloaded dictionaries for species composition and label mapping.

    Args:
        species_composition (Union[str, Dict[str, int]]):
            Either a path to a JSON file or a dictionary mapping species names to image counts.
        species_labels (Union[str, Dict[Union[int, str], str]]):
            Either a path to a JSON file or a dictionary mapping class labels to species names.

    Returns:
        torch.Tensor:
            A 1D tensor of weights (float32), one per class, ordered by class index.
            The weights are scaled so that their average equals 1.0.

    Notes:
        - Weights are computed using:
            `weight_i = (1 / count_i) / sum_j (1 / count_j) * num_classes`
        - This weighting compensates for class imbalance by amplifying loss contribution from rare classes.
        - Class order is determined by `species_labels_path` (i.e., label ID → name).
        - Assumes every species in the label mapping exists in the composition file.
    """
    if isinstance(species_composition, str):
        with open(species_composition, "r") as species_f:
            species_data = json.load(species_f)
    else:
        species_data = species_composition

    if isinstance(species_labels, str):
        with open(species_labels, "r") as labels_f:
            labels_data = json.load(labels_f)
    else:
        labels_data = species_labels

    species_names = list(labels_data.values())

    species_counts = []

    for species in species_names:
        species_counts.append(species_data[species])
    counts_tensor = torch.tensor(species_counts, dtype=torch.float)

    inv_freq = 1.0 / counts_tensor
    weights = inv_freq / inv_freq.sum() * len(inv_freq)
    return weights


def preprocess_eval_opencv(
    image_path, width, height, central_fraction=0.857, is_inception_v3: bool = False
):
    """
    Loads and preprocesses an image using OpenCV for inference with ONNX models.

    This function performs central cropping, resizing, normalization to [-1, 1], and converts the image
    to a CHW format suitable for ONNX inference.

    Args:
        image_path (str):
            Path to the image file to preprocess.
        width (int):
            Target width after resizing.
        height (int):
            Target height after resizing.
        central_fraction (float, optional):
            Fraction of the central region to retain before resizing. Defaults to 0.857.
        is_inception_v3 (bool, optional):
            If True, applies additional transposition to match InceptionV3's input format (HWC → CHW with batch first).
            Defaults to False.

    Returns:
        np.ndarray:
            A preprocessed image tensor of shape (1, 3, height, width) in float32 format, normalized to [-1, 1].

    Raises:
        ValueError:
            If the image cannot be read from the specified path.

    Notes:
        - Output is formatted as NCHW (batch size 1), suitable for ONNX runtime.
        - If `is_inception_v3=True`, an additional transpose is performed to produce (3, 1, H, W).
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError("Failed to read image")
    h, w, _ = img.shape

    crop_h = int(h * central_fraction)
    crop_w = int(w * central_fraction)

    offset_h = (h - crop_h) // 2
    offset_w = (w - crop_w) // 2
    img = img[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0

    if is_inception_v3:
        img = np.expand_dims(img, axis=0)
    else:
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))

    return img


import torch
import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from torch.utils.data import DataLoader
from pipeline.dataset_loader import CustomDataset
from pipeline.utility import (
    manifest_generator_wrapper,
    get_support_list,
    generate_report,
)
from typing import Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains the given model for one epoch on the provided DataLoader.

    This function performs standard supervised learning with forward pass, 
    loss computation, backpropagation, and optimizer updates. It also includes
    a label sanity check to ensure labels fall within valid class index range.

    Args:
        model (torch.nn.Module): 
            The model to be trained.
        dataloader (DataLoader[CustomDataset]): 
            A DataLoader that yields batches of (image, label) pairs.
        criterion: 
            A loss function (e.g., nn.CrossEntropyLoss).
        optimizer: 
            A PyTorch optimizer (e.g., torch.optim.Adam or SGD).
        device (torch.device): 
            The device on which to perform computation (CPU or GPU).

    Returns:
        Tuple[float, float]: 
            A tuple containing:
                - The average training loss over the entire dataset.
                - The training accuracy over the entire dataset.

    Notes:
        - A tensor guard checks that the ground-truth labels fall within the valid range `[0, num_classes - 1]` based on model output shape.
        - Accumulates total correct predictions and total loss for final reporting.
    """
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    checked_labels = False
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # tensor guard
        if not checked_labels:
            num_classes = model(images).shape[1]
            label_min = labels.min().item()
            label_max = labels.max().item()

            if labels.min() < 0 or labels.max() >= num_classes:
                raise ValueError(
                    f"Invalid labels detected!\n"
                    f"Labels: {labels}\n"
                    f"Min: {label_min}, Max: {label_max}\n"
                    f"Model output classes: {num_classes}"
                )
            checked_labels = True

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.detach().item():.3f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def sparse_warmup_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    optimizer,
    pruner,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    pruner.update_regularizer()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    checked_labels = False
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # tensor guard
        if not checked_labels:
            num_classes = model(images).shape[1]
            label_min = labels.min().item()
            label_max = labels.max().item()

            if labels.min() < 0 or labels.max() >= num_classes:
                raise ValueError(
                    f"Invalid labels detected!\n"
                    f"Labels: {labels}\n"
                    f"Min: {label_min}, Max: {label_max}\n"
                    f"Model output classes: {num_classes}"
                )
            checked_labels = True

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        pruner.regularize(model)
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.detach().item():.3f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def train_validate(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch with sparsity regularization applied via a pruner.

    This function is used during a warm-up phase of sparse training, where a regularizer (e.g., L1/L2 penalty on weights or activations) is applied to encourage sparsity before actual pruning is performed.

    Args:
        model (torch.nn.Module): 
            The model to be trained.
        dataloader (DataLoader[CustomDataset]): 
            A DataLoader that yields batches of (image, label) pairs.
        criterion: 
            The loss function used to train the model (e.g., CrossEntropyLoss).
        pruner: 
            A sparsity regularizer object that provides:
                - `update_regularizer()`: Called once before training.
                - `regularize(model)`: Called on each backward pass to apply regularization.
        device (torch.device): 
            The device on which to perform training (CPU or GPU).

    Returns:
        Tuple[float, float]: 
            A tuple containing:
                - Average loss over the epoch.
                - Accuracy over the entire dataset.
    """
    model.eval()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Validating", unit="batch", leave=False)
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.detach().item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    return avg_loss, accuracy, float(macro_f1)


def save_model(
    model: torch.nn.Module,
    name: str,
    save_path: str,
    device: torch.device,
    img_size: Tuple[int, int],
):
    """
    Saves a PyTorch model in both `.pth` and ONNX formats.

    This function exports the given model to:
        - PyTorch format (.pth) using `torch.save()`
        - ONNX format (.onnx) using `torch.onnx.export()`, with support for dynamic batch sizes

    Args:
        model (torch.nn.Module): 
            The trained PyTorch model to be saved.
        name (str): 
            Base name for the output files (e.g., 'mobilenetv3').
        save_path (str): 
            Directory where the model files will be saved. Will be created if it doesn't exist.
        device (torch.device): 
            Device on which to create the dummy input tensor for ONNX export.
        img_size (Tuple[int, int]): 
            Expected input image size as (height, width) for dummy input.

    Output Files:
        - `<save_path>/<name>.pth`: PyTorch serialized model.
        - `<save_path>/<name>.onnx`: ONNX exported model with dynamic batch dimension.

    Notes:
        - ONNX export uses `opset_version=14` and includes constant folding for optimization.
        - Assumes the model expects input shape `(N, 3, H, W)` where H and W are from `img_size`.
        - Dynamic axes allow for variable batch sizes during ONNX inference.
    """
    os.makedirs(save_path, exist_ok=True)
    pytorch_path = os.path.join(save_path, f"{name}.pth")
    torch.save(model, pytorch_path)
    print(f"Saved Pytorch model to {pytorch_path}")

    dummy_input = torch.randn(1, 3, *img_size, device=device)
    onnx_path = os.path.join(save_path, f"{name}.onnx")
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {onnx_path}")


def validation_onnx(onnx_path: str, device: str = "cpu"):
    """
    Validates an ONNX model against a given validation dataset and computes classification metrics.

    This function loads an ONNX model, runs inference on a validation DataLoader, collects predictions, and evaluates performance using accuracy, weighted recall, and weighted F1-score.
    It also generates a per-class evaluation report as a pandas DataFrame.

    Args:
        onnx_path (str): 
            Path to the ONNX model file.
        device (str, optional): 
            Device to use for inference ("cpu" or "cuda").

    Returns:
        pd.DataFrame: 
            A classification report containing metrics such as precision, recall, and F1-score
            for each species class.

    Notes:
        - Assumes input tensors are normalized and shaped as (N, 3, H, W) and converts them to (N, H, W, 3) for ONNX runtime.
        - Dynamically detects the dominant threshold from the ONNX model filename (expects format like `model_80.onnx`).
    """
    model_name = os.path.basename(onnx_path)
    dom = model_name.replace(".onnx", "").split("_")[1]
    _, train_images, val_images, species_dict, species_composition = manifest_generator_wrapper(1.0)
    val_dataset = CustomDataset(val_images, train=False, img_size=(224, 224))
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    species_names = list(species_dict.values())
    total_support_list = get_support_list(species_composition, species_names)

    ort_session = ort.InferenceSession(
        onnx_path,
        providers=[
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        ],
    )
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    all_preds, all_labels = [], []

    for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
        # Convert tensor to numpy + permute to NHWC
        # images_np = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        images_np = images.cpu().numpy().astype(np.float32)

        outputs = ort_session.run([output_name], {input_name: images_np})[0]
        preds = np.argmax(outputs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"ONNX Validation Accuracy: {accuracy:.4f}")
    print(f"ONNX Weighted Recall: {recall:.4f}")
    print(f"ONNX Weighted F1-Score: {f1:.4f}")

    report_df = generate_report(
        all_labels, all_preds, species_names, total_support_list, float(accuracy)
    )
    return report_df
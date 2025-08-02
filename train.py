import time
import argparse
from typing import Dict, Any
import torch
from torch.optim import Adam
from mcunet.model_zoo import build_model
from utilities import get_device
from models.model_handler import ModelHandler
from data_prep.data_loader import DataLoaderCreator
from models.convnext_model import ConvNeXt160


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments for the training script.

    Returns:
        Dict[str, Any]: Dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train MCUNet models for TinyML applications"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="mcunet-in2",
        choices=[
            "mcunet-in1",
            "mcunet-in2",
            "mcunet-in4",
            "mcunet-in5",
            "mcunet-in6",
            "convnext-large",
            "convnext-tiny",
            "convnext-base",
        ],
        help="MCUNet model variant (default: mcunet-in2)",
    )

    # Dataset parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Dominance threshold for dataset selection (default: 0.2)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[160, 160],
        help="Image size for training (height, width) (default: 160 160)",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)",
    )

    parser.add_argument(
        "--start_rank",
        type=int,
        default=0,
        help="Starting rank for distributed training (default: 0)",
    )
    # Parse the arguments
    args = parser.parse_args()
    return vars(args)  # Convert to dictionary


# Now replace your hardcoded parameters with the command-line arguments
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Training parameters
    BATCH_SIZE = args["batch_size"]
    NUM_WORKERS = args["workers"]
    NUM_EPOCHS = args["epochs"]
    LR = args["lr"]

    # Model parameters
    MODEL_NAME = args["model"]
    IMG_SIZE = tuple(args["img_size"])
    OUTPUT_DIR = args["output_dir"]

    # Dataset parameters
    DOMINANCE_THRESHOLD = args["threshold"]
    START_RANK = args["start_rank"]

    print("Training with parameters:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Learning Rate: {LR}")
    print(f"  Dominance Threshold: {DOMINANCE_THRESHOLD}")

    data_loader = DataLoaderCreator(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dominant_threshold=DOMINANCE_THRESHOLD,
        start_rank=START_RANK,
    )
    train_loader, val_loader, weights, NUM_SPECIES, small_species_label = (
        data_loader.create_dataloader()
    )

    # convert weights from dict to tensor
    if weights is not None:
        weights_tensor = torch.tensor(
            [weights.get(i, 1.0) for i in range(NUM_SPECIES)],
            dtype=torch.float32,
        )
    else:
        weights_tensor = None
    print(f"weights tensor: {weights_tensor}")
    if MODEL_NAME == "convnext-tiny":
        # Use ConvNeXt model
        # Define model
        model = ConvNeXt160(
            num_classes=NUM_SPECIES, pretrained=True, convext_ver="convnext_tiny"
        )
    elif MODEL_NAME == "convnext-base":
        # Use ConvNeXt model
        # Define model
        model = ConvNeXt160(
            num_classes=NUM_SPECIES, pretrained=True, convext_ver="convnext_base"
        )
    elif MODEL_NAME == "convnext-large":
        # Use ConvNeXt model
        # Define model
        model = ConvNeXt160(
            num_classes=NUM_SPECIES, pretrained=True, convext_ver="convnext_large"
        )

    # if MODEL_NAME == "convnext-large":
    #     model = convnext_large_builder(
    #         device, num_outputs=NUM_SPECIES, start_with_weight=True, input_size=IMG_SIZE
    #     )

    else:
        # Build model - IMPORTANT: Use TOTAL_CLASSES instead of NUM_SPECIES
        model, image_size, description = build_model(net_id=MODEL_NAME, pretrained=True)
        in_features = model.classifier.linear.in_features  # type : ignore
        model.classifier.linear = torch.nn.Linear(  # type: ignore
            in_features, NUM_SPECIES
        )  # Changed this line

    device = get_device()
    print(f"Model configured for {NUM_SPECIES} classes")
    # Calculating weight for criterion for imbalanced dataset
    class_weights = torch.tensor(
        [1.0] * NUM_SPECIES, dtype=torch.float32
    )  # Placeholder for class weights
    # Set up training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # if MODEL_NAME == "convnext-large":
    #     model = model
    # else:
    model.to(device)
    model_handler = ModelHandler(device)

    # Training loop
    best_acc = 0.0
    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        train_loss, train_acc = model_handler.train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, macro_f1 = model_handler.train_validate(
            model, val_loader, criterion
        )
        scheduler.step()
        print(
            f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val acc: {val_acc:.4f} Val F1: {macro_f1:.4f}"
        )
        if macro_f1 > best_f1:
            start_save = time.perf_counter()
            best_acc = val_acc
            best_f1 = macro_f1
            print(
                f"Saving model with accuracy: {best_acc:.4f} and F1-score: {best_f1:.4f}"
            )
            name = f"{MODEL_NAME}_haute_garonne_{DOMINANCE_THRESHOLD}_{START_RANK}_best"
            model_handler.save_model(model, name, OUTPUT_DIR, IMG_SIZE)
            end_save = time.perf_counter()
            print(f"Save time: {end_save - start_save:.2f}s")
        end = time.perf_counter()
        print(f"Total time: {end - start:.2f}s")
    print(f"Best accuracy: {best_acc} with F1-score: {best_f1}")

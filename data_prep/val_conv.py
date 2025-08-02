import pandas as pd
import torch  # type: ignore
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from pipeline.utility import (
    get_device,
    get_support_list,
    generate_report,
    convnext_large_builder,
    manifest_generator_wrapper,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score
from pipeline.dataset_loader import CustomDataset
import time

# NNUM_SPECIES = 1000  # Adjust this based on your dataset
def validate_convnext_model(
    model_path: str = "./models/convnext_full_insect_best.pth",
    batch_size: int = 1,
    num_workers: int = 1,
    report_path: str = "./test.csv"
):
    device = get_device()
    NAME = "convnext_full_insect"

    _, _, val_images, species_labels, species_composition = manifest_generator_wrapper(1.0)

    species_names = list(species_labels.values())
    val_dataset = CustomDataset(val_images, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    total_support_list = get_support_list(species_composition, species_names)

    model = convnext_large_builder(
        device,
        num_outputs=len(species_names),
        start_with_weight=True,
        path=model_path,
    )
    val_loss, val_correct = 0.0, 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    print("Begin validating")

    # Timing measurement
    total_images = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
            images = images.to(device)
            batch_size = images.size(0)
            total_images += batch_size

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = total_images / total_time if total_time > 0 else 0

    accuracy = accuracy_score(all_labels, all_preds)
    weighted_recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted Average F1-Score: {f1:.4f}")
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f}s, FPS (throughput): {fps:.2f}")

    report_df = generate_report(
        all_labels, all_preds, species_names, total_support_list, float(accuracy)
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        report_df.to_csv(report_path)

    return {
        "accuracy": accuracy,
        "weighted_recall": weighted_recall,
        "f1": f1,
        "total_images": total_images,
        "total_time": total_time,
        "fps": fps,
        "report_df": report_df,
    }

scores = validate_convnext_model(
    model_path="models/convnext_full_insect_best.pth",
    batch_size=1,
    num_workers=1,
    report_path="./test.csv"
)
print(scores)
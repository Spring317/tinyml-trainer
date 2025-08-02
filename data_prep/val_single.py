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
import mcunet
import time

from val_conv import validate_convnext_model


# NNUM_SPECIES = 1000  # Adjust this based on your dataset
def eval_single(device, BATCH_SIZE, NUM_WORKERS, CLASS):
    _, _, val_images, species_labels, species_composition = manifest_generator_wrapper(
        1.0
    )
    species_names = list(species_labels.values())
    val_dataset = CustomDataset(val_images, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    # if CLASS == 1:
    #     print("Using only model2")
    #     scores =validate_convnext_model(
    #         model_path="models/convnext_full_insect_best.pth",
    #         batch_size=BATCH_SIZE,
    #         num_workers=NUM_WORKERS,
    #         report_path="./test.csv"
    #     )
    #     return scores

    total_support_list = get_support_list(species_composition, species_names)
    model1 = torch.load(
        f"models/mcunet-in2-haute-garonne_{CLASS}_best.pth",
        map_location=device,
        weights_only=False,
    )

    model2 = convnext_large_builder(
        device,
        num_outputs=len(species_names),
        start_with_weight=True,
        path="models/convnext_full_insect_best.pth",
    )
    model1.eval()
    model2.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    # Timing and counting
    total_images = 0
    model1_time = 0.0
    model2_time = 0.0
    model1_calls = 0
    model2_calls = 0
    start_time = time.perf_counter()

    print("Begin validating")

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
            images = images.to(device)
            batch_size = images.size(0)
            total_images += batch_size

            t1 = time.perf_counter()
            outputs = model1(images)
            t2 = time.perf_counter()
            model1_time += t2 - t1
            model1_calls += 1

            _, preds = torch.max(outputs, 1)
            if preds.item() == CLASS - 1:
                t3 = time.perf_counter()
                outputs = model2(images)
                t4 = time.perf_counter()
                model2_time += t4 - t3
                model2_calls += 1
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = total_images / total_time if total_time > 0 else 0
    avg_model1_time = model1_time / model1_calls if model1_calls else 0
    avg_model2_time = model2_time / model2_calls if model2_calls else 0
    model1_fps = 1.0 / avg_model1_time if avg_model1_time > 0 else 0
    model2_fps = 1.0 / avg_model2_time if avg_model2_time > 0 else 0

    accuracy = accuracy_score(all_labels, all_preds)
    weighted_recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted Average F1-Score: {f1:.4f}")
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f}s, FPS: {fps:.2f}")
    print(
        f"Model1 calls: {model1_calls}, avg time: {avg_model1_time * 1000:.2f} ms, FPS: {model1_fps:.2f}"
    )
    print(
        f"Model2 calls: {model2_calls}, avg time: {avg_model2_time * 1000:.2f} ms, FPS: {model2_fps:.2f}"
    )

    report_df = generate_report(
        all_labels, all_preds, species_names, total_support_list, float(accuracy)
    )

    # Save all metrics to CSV
    metrics = {
        "class": CLASS,
        "total_images": total_images,
        "total_time_sec": total_time,
        "overall_fps": fps,
        "model1_calls": model1_calls,
        "model1_avg_time_ms": avg_model1_time * 1000,
        "model1_fps": model1_fps,
        "model2_calls": model2_calls,
        "model2_avg_time_ms": avg_model2_time * 1000,
        "model2_fps": model2_fps,
        "accuracy": accuracy,
        "weighted_recall": weighted_recall,
        "weighted_f1": f1,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"./metrics_{CLASS}.csv", index=False)
    report_df.to_csv(f"./test_{CLASS}.csv")
    return metrics


if __name__ == "__main__":
    classes = [3, 5, 8]
    all_metrics = []
    for c in classes:
        metrics = eval_single(
            device=get_device(),
            BATCH_SIZE=1,
            NUM_WORKERS=1,
            CLASS=c,
        )

        all_metrics.append(metrics)
    pd.DataFrame(all_metrics).to_csv("./metrics_summary.csv", index=False)

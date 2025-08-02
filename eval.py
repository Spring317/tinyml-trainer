import os
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import time
from utility.utilities import get_device
from utility.plots import (
    plot_confusion_matrix_heatmap,
    plot_class_f1_scores,
    plot_class_precision_recall,
)
from sklearn.metrics import confusion_matrix
from utility.scores import (
    get_classification_report,
    calculate_scores,
    calculate_other_precision_recall_f1,
)
from data_prep.data_loader import DataLoaderCreator
from data_prep.class_handler import get_class_info_for_evaluation


def evaluate_model(
    model_path: str,
    dominant_threshold: float,
    start_rank: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = "evaluation_results",
    measure_timing: bool = False,
    number_of_dominant_classes: int = 3,
):
    """
    Comprehensive evaluation of a trained model with class-specific metrics and timing.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and dataset
    device = get_device()
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    # Get class information
    class_names, num_classes, class_to_idx = get_class_info_for_evaluation(
        start_rank=start_rank,
        number_of_dominant_classes=number_of_dominant_classes,  # Default to 3 main classes + 1 "Other"
        model_path=model_path,
    )

    print(f"Loaded model from {model_path}")
    print(f"Evaluating with {num_classes} classes: {class_names}")

    if measure_timing:
        print(f" Timing mode enabled - batch_size={batch_size}, device={device}")

    # Create dataloader
    dataloader_creator = DataLoaderCreator(
        batch_size, num_workers, dominant_threshold, start_rank
    )

    # Get validation dataloader
    _, val_loader, _, _, _ = dataloader_creator.create_dataloader()

    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []

    # Timing variables
    inference_times = []
    total_samples = 0

    if measure_timing:
        print("🔥 Warming up GPU...")
        # GPU warmup
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, 3, 160, 160).to(device)
            for _ in range(10):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        print("✅ GPU warmed up!")

    print("Starting evaluation...")

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(
            tqdm(val_loader, desc="Evaluating")
        ):
            images = images.to(device)
            labels = labels.to(device)

            if measure_timing:
                # Synchronize GPU before timing
                if device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.time()

            # Model inference
            outputs = model(images)

            if measure_timing:
                # Synchronize GPU after inference
                if device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()

                # Calculate timing
                batch_inference_time = end_time - start_time
                current_batch_size = images.shape[0]
                per_sample_time = batch_inference_time / current_batch_size

                # Store timing for each sample in batch
                inference_times.extend([per_sample_time] * current_batch_size)
                total_samples += current_batch_size

                # Print progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    avg_time = np.mean(inference_times[-current_batch_size * 100 :])
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(
                        f"Batch {batch_idx + 1}: Avg time/sample: {avg_time * 1000:.2f}ms, FPS: {fps:.1f}"
                    )

            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            # Collect results
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    count = 0
    for i in range(len(all_preds)):
        if all_preds[i] == all_labels[i]:
            count += 1

    print(f"Total correct predictions: {count}/{len(all_preds)}")
    # Calculate timing statistics if timing was measured
    timing_stats = {}
    if measure_timing and inference_times:
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        std_inference_time = np.std(inference_times) * 1000
        min_inference_time = np.min(inference_times) * 1000
        max_inference_time = np.max(inference_times) * 1000
        median_inference_time = np.median(inference_times) * 1000

        # Calculate FPS and throughput
        avg_fps = 1.0 / np.mean(inference_times)
        throughput = avg_fps  # samples per second

        timing_stats = {
            "avg_inference_time_ms": avg_inference_time,
            "std_inference_time_ms": std_inference_time,
            "min_inference_time_ms": min_inference_time,
            "max_inference_time_ms": max_inference_time,
            "median_inference_time_ms": median_inference_time,
            "avg_fps": avg_fps,
            "throughput_samples_per_sec": throughput,
            "total_samples": total_samples,
            "batch_size": batch_size,
        }

        print(f"\n⏱️ TIMING PERFORMANCE:")
        print(
            f"  • Average inference time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms"
        )
        print(
            f"  • Min/Max inference time: {min_inference_time:.2f} / {max_inference_time:.2f} ms"
        )
        print(f"  • Median inference time: {median_inference_time:.2f} ms")
        print(f"  • Average FPS: {avg_fps:.1f}")
        print(f"  • Throughput: {throughput:.1f} samples/second")
        print(f"  • Total samples processed: {total_samples}")

        # Save detailed timing data
        timing_df = pd.DataFrame(
            {
                "sample_index": range(len(inference_times)),
                "inference_time_ms": np.array(inference_times) * 1000,
                "fps": 1.0 / np.array(inference_times),
            }
        )
        timing_df.to_csv(f"{output_dir}/timing_details.csv", index=False)

        # Save timing summary
        timing_summary_df = pd.DataFrame([timing_stats])
        timing_summary_df.to_csv(f"{output_dir}/timing_summary.csv", index=False)

    # Calculate overall metrics
    overall_accuracy, overall_precision, overall_recall, overall_f1 = calculate_scores(
        all_labels,
        all_preds,
    )

    print("\n===== OVERALL METRICS =====")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision (macro): {overall_precision:.4f}")
    print(f"Recall (macro): {overall_recall:.4f}")
    print(f"F1 Score (macro): {overall_f1:.4f}")

    # Get detailed classification report
    report = get_classification_report(
        all_labels,
        all_preds,
        num_classes=num_classes,
        class_names=class_names,
        output_dict=True,
    )

    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Calculate precision and recall for the "Other" class if it exists
    if num_classes > 3:  # Assuming last class is "Other"
        pred, rec, f1 = calculate_other_precision_recall_f1(
            all_labels, all_preds, num_classes
        )
        other_class_name = class_names[-1]  # Last class should be "Other"
        report_df.loc[other_class_name, "precision"] = pred
        report_df.loc[other_class_name, "recall"] = rec
        report_df.loc[other_class_name, "f1-score"] = f1

    # Save classification report
    report_df.to_csv(f"{output_dir}/classification_report.csv")

    # Print class-specific metrics
    print("\n===== CLASS-SPECIFIC METRICS =====")
    class_metrics = []

    for i in range(num_classes):
        class_name = class_names[i]
        key = str(i)

        # Check if the key exists in the report, otherwise try class_name
        if key not in report:
            key = class_name
            if key not in report:
                print(f"Warning: Could not find metrics for class {i} ({class_name})")
                continue

        metrics = {
            "class_name": class_name,
            "precision": report[key]["precision"],
            "recall": report[key]["recall"],
            "f1_score": report[key]["f1-score"],
            "support": report[key]["support"],
            "accuracy": report[key].get("accuracy", 0.0),
        }
        class_metrics.append(metrics)

        print(f"Class: {class_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Support: {metrics['support']}")

    # Create DataFrame and save to CSV
    class_df = pd.DataFrame(class_metrics)
    class_df.to_csv(f"{output_dir}/class_metrics.csv", index=False)

    # Calculate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")

    # Calculate per-class confidence scores
    class_confidence = []
    for i in range(num_classes):
        pred_class_i = all_preds == i
        if np.sum(pred_class_i) > 0:
            conf_scores = all_probs[pred_class_i, i]
            class_confidence.append(
                {
                    "class_name": class_names[i],
                    "mean_confidence": np.mean(conf_scores),
                    "min_confidence": np.min(conf_scores),
                    "max_confidence": np.max(conf_scores),
                    "std_confidence": np.std(conf_scores),
                }
            )

    # Save confidence scores
    if class_confidence:
        confidence_df = pd.DataFrame(class_confidence)
        confidence_df.to_csv(f"{output_dir}/class_confidence.csv", index=False)

    # Generate plots
    plot_confusion_matrix_heatmap(cm, class_names, output_dir)
    plot_class_f1_scores(class_df, output_dir)
    plot_class_precision_recall(class_df, output_dir)

    print(f"\nEvaluation complete. Results saved to {output_dir}/")

    result = {
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "class_metrics": class_df,
        "confusion_matrix": cm_df,
    }

    # Add timing stats to result if measured
    if timing_stats:
        result["timing_stats"] = timing_stats

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model with detailed metrics and timing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/mcunet_haute_garonne_8_species.pth",
        help="Path to model file (.pth)",
    )
    parser.add_argument(
        "--dominant_threshold",
        type=float,
        default=0.5,
        help="Threshold for dominant species classification",
    )
    parser.add_argument(
        "--start_rank", type=int, default=0, help="Starting rank for dataset creation"
    )
    parser.add_argument(
        "--number_of_dominant_classes",
        type=int,
        default=3,
        help="Number of dominant classes to consider (default: 3 main classes + 1 'Other')",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable detailed timing measurements (use batch_size=1 for accurate per-sample timing)",
    )

    args = parser.parse_args()

    # Suggest optimal settings for timing
    if args.timing and args.batch_size > 1:
        print("⚠️  For most accurate timing, consider using --batch_size 1")

    if args.timing and args.num_workers > 0:
        print("⚠️  For most accurate timing, consider using --num_workers 0")

    evaluate_model(
        model_path=args.model,
        dominant_threshold=args.dominant_threshold,
        start_rank=args.start_rank,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output,
        measure_timing=args.timing,
        number_of_dominant_classes=args.number_of_dominant_classes,
    )

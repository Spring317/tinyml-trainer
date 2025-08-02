# Add these imports at the top
import onnxruntime as ort
import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDataset import CustomDataset
from utilities import get_device
# Add a new function to evaluate ONNX models
def evaluate_onnx_model(model_path, val_dataset_path, batch_size=32, num_workers=4, output_dir="evaluation_results"):
    """
    Comprehensive evaluation of a quantized ONNX model with detailed metrics.
    
    Args:
        model_path (str): Path to the ONNX model
        val_dataset_path (str): Path to the validation dataset
        batch_size (int): Batch size for evaluation
        num_workers (int): Number of workers for data loading
        output_dir (str): Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ONNX model
    print(f"Loading ONNX model from {model_path}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
    
    # Get input/output details
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Determine image size from model input
    if len(input_shape) == 4:  # [batch, channels, height, width]
        img_height = input_shape[2] if input_shape[2] != -1 else 224
        img_width = input_shape[3] if input_shape[3] != -1 else 224
        channels = input_shape[1]
    else:
        # Default to common image sizes if we can't determine
        img_height, img_width = 224, 224
        channels = 3
        
    print(f"Model expects input shape: {input_shape}, using size: {img_height}x{img_width}x{channels}")
    
    # Load dataset
    val_dataset = torch.load(val_dataset_path, weights_only=False)
    
    # Get class names using the same approach as before
    try:
        # First attempt: check if the dataset has a class_to_idx attribute
        class_to_idx = getattr(val_dataset, 'class_to_idx', None)
        if class_to_idx:
            class_names = list(class_to_idx.keys())
        else:
            # Second attempt: check for species_labels attribute
            species_labels = getattr(val_dataset, 'species_labels', None)
            if species_labels:
                class_names = list(species_labels.keys())
            else:
                # Third attempt: check for classes attribute
                classes = getattr(val_dataset, 'classes', None)
                if classes:
                    class_names = classes
                else:
                    # Fallback: use numeric class labels
                    all_labels_set = set(val_dataset.tensors[1].numpy() if hasattr(val_dataset, 'tensors') else 
                                        [label for _, label in val_dataset])
                    class_names = [f"Class {i}" for i in sorted(all_labels_set)]
    except Exception as e:
        print(f"Warning: Could not extract class names automatically: {e}")
        print("Using numeric class indices instead.")
        # Extract unique labels from the dataset
        all_labels = []
        for _, label in val_dataset:
            all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        unique_labels = sorted(set(all_labels))
        class_names = [f"Class {i}" for i in unique_labels]
    
    num_classes = len(class_names)
    
    print(f"Loaded validation dataset with {len(val_dataset)} samples and {num_classes} classes")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []  # For confidence scores
    inference_times = []  # Track inference times
    
    # Evaluate model
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        # Convert images to numpy with correct format
        # ONNX models typically expect NCHW format (batch, channels, height, width)
        batch_images = images.numpy()
        
        # Time the inference
        start_time = time.time()
        
        # Run ONNX inference
        outputs = session.run([output_name], {input_name: batch_images})[0]
        
        # Record inference time
        inference_times.append((time.time() - start_time) * 1000)  # in ms
        
        # Convert outputs to predictions and probabilities
        probs = softmax(outputs, axis=1)
        preds = np.argmax(outputs, axis=1)
        
        # Store results
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    
    # Convert to numpy arrays if not already
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Print inference time statistics
    avg_inference_time = np.mean(inference_times)
    print(f"\n===== INFERENCE PERFORMANCE =====")
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Min/Max inference time: {np.min(inference_times):.2f}/{np.max(inference_times):.2f} ms")
    print(f"95th percentile: {np.percentile(inference_times, 95):.2f} ms")
    
    # Save inference time data
    inference_df = pd.DataFrame({"inference_time_ms": inference_times})
    inference_df.to_csv(f"{output_dir}/inference_times.csv", index=False)
    
    # Plot inference time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=30)
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Inference Time Distribution')
    plt.axvline(avg_inference_time, color='r', linestyle='dashed', linewidth=1)
    plt.text(avg_inference_time*1.05, plt.ylim()[1]*0.9, f'Mean: {avg_inference_time:.2f} ms')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/inference_time_distribution.png", dpi=300)
    
    # Rest of the evaluation code remains mostly the same...
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds, average='macro')
    overall_recall = recall_score(all_labels, all_preds, average='macro')
    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n===== OVERALL METRICS =====")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision (macro): {overall_precision:.4f}")
    print(f"Recall (macro): {overall_recall:.4f}")
    print(f"F1 Score (macro): {overall_f1:.4f}")
    
    # Get detailed classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Print the report keys for debugging
    print(f"Report keys: {list(report.keys())}")
    
    # Convert classification report to DataFrame for easier handling
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_df.to_csv(f"{output_dir}/classification_report.csv")
    
    # Create a DataFrame for class metrics
    class_metrics = []
    for i in range(num_classes):
        class_name = class_names[i]
        # Handle different key formats in the classification report
        key = str(i)
        
        # Check if the key exists in the report, otherwise try class_name
        if key not in report:
            key = class_name
            # If that doesn't work either, skip this class
            if key not in report:
                print(f"Warning: Could not find metrics for class {i} ({class_name}) in the report")
                print(f"Available keys in report: {list(report.keys())}")
                continue
        
        metrics = {
            'class_name': class_name,
            'precision': report[key]['precision'],
            'recall': report[key]['recall'],
            'f1_score': report[key]['f1-score'],
            'support': report[key]['support'],
            'accuracy': accuracy_score(all_labels == i, all_preds == i),
        }
        class_metrics.append(metrics)
        
        print(f"Class: {class_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
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
        # Get samples where the model predicted class i
        pred_class_i = all_preds == i
        if np.sum(pred_class_i) > 0:
            # Get the confidence scores for those predictions
            conf_scores = np.array([probs[i] for probs, pred in zip(all_probs, all_preds) if pred == i])
            class_confidence.append({
                'class_name': class_names[i],
                'mean_confidence': np.mean(conf_scores),
                'min_confidence': np.min(conf_scores),
                'max_confidence': np.max(conf_scores),
                'std_confidence': np.std(conf_scores)
            })
    
    # Save confidence scores
    confidence_df = pd.DataFrame(class_confidence)
    confidence_df.to_csv(f"{output_dir}/class_confidence.csv", index=False)
    
    # Generate plots
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    
    # 2. Class F1 Scores
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x='class_name', y='f1_score', data=class_df)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_f1_scores.png", dpi=300)
    
    # 3. Precision & Recall by Class
    plt.figure(figsize=(14, 6))
    class_df_melted = pd.melt(class_df, id_vars=['class_name'], value_vars=['precision', 'recall'], var_name='metric', value_name='score')
    sns.barplot(x='class_name', y='score', hue='metric', data=class_df_melted)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision and Recall by Class')
    plt.xticks(rotation=90)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall.png", dpi=300)
    
    # 4. Class Confidence Distribution
    if len(class_confidence) > 0:
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x='class_name', y='mean_confidence', data=confidence_df)
        plt.errorbar(
            x=np.arange(len(confidence_df)),
            y=confidence_df['mean_confidence'],
            yerr=confidence_df['std_confidence'],
            fmt='none',
            capsize=5,
            color='black'
        )
        plt.xlabel('Class')
        plt.ylabel('Mean Confidence Score')
        plt.title('Mean Prediction Confidence by Class')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_confidence.png", dpi=300)
    
    save_comprehensive_results(
        output_dir=output_dir,
        all_preds=all_preds,
        all_labels=all_labels,
        all_probs=all_probs,
        inference_times=inference_times,
        class_names=class_names
    )

    return {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_metrics': class_df,
        'confusion_matrix': cm_df
    }

def save_comprehensive_results(output_dir, all_preds, all_labels, all_probs, inference_times, class_names):
    """
    Save comprehensive evaluation results to CSV files for further analysis.
    
    Args:
        output_dir: Directory where to save result files
        all_preds: Array of all predictions
        all_labels: Array of all ground truth labels
        all_probs: Array of all probability outputs
        inference_times: List of inference times for each batch
        class_names: Names of all classes
    """
    # 1. Save per-sample predictions and ground truth
    sample_results = []
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        probs = all_probs[i]
        sample_result = {
            'sample_id': i,
            'true_label': label,
            'true_class': class_names[label] if label < len(class_names) else f"Unknown-{label}",
            'predicted_label': pred,
            'predicted_class': class_names[pred] if pred < len(class_names) else f"Unknown-{pred}",
            'correct': int(pred == label),
        }
        
        # Add probability for each class
        for j, class_name in enumerate(class_names):
            sample_result[f'prob_{class_name}'] = probs[j]
            
        sample_results.append(sample_result)
    
    sample_df = pd.DataFrame(sample_results)
    sample_df.to_csv(f"{output_dir}/per_sample_results.csv", index=False)
    print(f"Saved per-sample results to {output_dir}/per_sample_results.csv")
    
    # 2. Save detailed inference time statistics
    inference_stats = {
        'count': len(inference_times),
        'mean_ms': np.mean(inference_times),
        'std_ms': np.std(inference_times),
        'min_ms': np.min(inference_times),
        'max_ms': np.max(inference_times),
        'median_ms': np.median(inference_times),
        'p25_ms': np.percentile(inference_times, 25),
        'p75_ms': np.percentile(inference_times, 75),
        'p90_ms': np.percentile(inference_times, 90),
        'p95_ms': np.percentile(inference_times, 95),
        'p99_ms': np.percentile(inference_times, 99),
        'iqr_ms': np.percentile(inference_times, 75) - np.percentile(inference_times, 25),
    }
    
    infer_stats_df = pd.DataFrame([inference_stats])
    infer_stats_df.to_csv(f"{output_dir}/inference_statistics.csv", index=False)
    print(f"Saved inference statistics to {output_dir}/inference_statistics.csv")
    
    # 3. Save error analysis - samples with incorrect predictions
    errors = sample_df[sample_df['correct'] == 0].copy()
    if len(errors) > 0:
        # Add confidence margin (difference between top predicted class and true class)
        for i, row in errors.iterrows():
            true_label = row['true_label']
            pred_label = row['predicted_label']
            true_prob = row[f'prob_{class_names[true_label]}'] if true_label < len(class_names) else 0
            pred_prob = row[f'prob_{class_names[pred_label]}'] if pred_label < len(class_names) else 0
            errors.loc[i, 'confidence_margin'] = pred_prob - true_prob
            
        errors.to_csv(f"{output_dir}/misclassifications.csv", index=False)
        print(f"Saved {len(errors)} misclassifications to {output_dir}/misclassifications.csv")
    
    # 4. Save per-class statistics
    class_stats = []
    for i, class_name in enumerate(class_names):
        # Calculate metrics for this class
        true_positives = np.sum((all_labels == i) & (all_preds == i))
        false_positives = np.sum((all_labels != i) & (all_preds == i))
        false_negatives = np.sum((all_labels == i) & (all_preds != i))
        true_negatives = np.sum((all_labels != i) & (all_preds != i))
        
        # Calculate derived metrics
        total_samples = len(all_labels)
        class_samples = np.sum(all_labels == i)
        class_predictions = np.sum(all_preds == i)
        
        # Get confidence stats for correctly classified samples
        correct_indices = (all_labels == i) & (all_preds == i)
        if np.any(correct_indices):
            correct_probs = np.array([p[i] for p, correct in zip(all_probs, correct_indices) if correct])
            mean_confidence = np.mean(correct_probs) if len(correct_probs) > 0 else 0
            min_confidence = np.min(correct_probs) if len(correct_probs) > 0 else 0
            max_confidence = np.max(correct_probs) if len(correct_probs) > 0 else 0
        else:
            mean_confidence = min_confidence = max_confidence = 0
            
        # Add all statistics
        class_stats.append({
            'class_name': class_name,
            'class_id': i,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
            'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
            'specificity': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0,
            'f1_score': 2 * true_positives / (2 * true_positives + false_positives + false_negatives) if (2 * true_positives + false_positives + false_negatives) > 0 else 0,
            'accuracy': (true_positives + true_negatives) / total_samples,
            'support': class_samples,
            'mean_confidence': mean_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'fraction_of_dataset': class_samples / total_samples if total_samples > 0 else 0,
            'model_preference': class_predictions / total_samples if total_samples > 0 else 0,
        })
    
    class_stats_df = pd.DataFrame(class_stats)
    class_stats_df.to_csv(f"{output_dir}/detailed_class_metrics.csv", index=False)
    print(f"Saved detailed per-class statistics to {output_dir}/detailed_class_metrics.csv")
    
    # 5. Save raw inference times for distribution analysis
    pd.DataFrame({'time_ms': inference_times}).to_csv(
        f"{output_dir}/raw_inference_times.csv", index=False
    )
    
    # 6. Save confusion matrix with both absolute and percentage values
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{output_dir}/confusion_matrix_absolute.csv")
    
    # Calculate percentage per row (recall perspective)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_pct = np.nan_to_num(cm_pct * 100.0, nan=0.0)  # Convert NaN to 0
    cm_pct_df = pd.DataFrame(cm_pct, index=class_names, columns=class_names)
    cm_pct_df.to_csv(f"{output_dir}/confusion_matrix_percent.csv")
    
    print(f"All evaluation results have been saved to {output_dir}/")

# Add a softmax implementation since we're working with raw logits from ONNX
def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Update the main block to include ONNX support
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model with detailed metrics')
    parser.add_argument('--model', type=str, default="models/mcunet_haute_garonne_8_species_quantized.onnx",
                        help='Path to model file (.pth or .onnx)')
    parser.add_argument('--onnx', action='store_true', help='Evaluate ONNX model instead of PyTorch model')
    parser.add_argument('--dataset', type=str, default="val_dataset.pt", help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--output', type=str, default="evaluation_results", help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.onnx or args.model.endswith('.onnx'):
        # For ONNX models
        import time  # Add time import for measuring inference speed
        evaluate_onnx_model(
            model_path=args.model,
            val_dataset_path=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output
        )
    else:
        # For PyTorch models
        evaluate_model(
            model_path=args.model,
            val_dataset_path=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output
        )
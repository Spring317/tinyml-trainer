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
from train_bin import BinaryInsectDataset
from CustomDataset import CustomDataset
from utilities import get_device, manifest_generator_wrapper

def calculate_other_precision_recall_f1(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Last class index
    i = num_classes - 1

    # Extract TP, FP, FN for the last class
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    
    # Compute modified precision and recall
    mod_precision = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    mod_recall = FP / (FP + TN) if (TP + FN) > 0 else 0.0
    mod_f1 = 2 * (mod_precision * mod_recall) / (mod_precision + mod_recall) if (mod_precision + mod_recall) > 0 else 0.0

    return mod_precision, mod_recall, mod_f1

def evaluate_model(model_path, dominant_threshold, batch_size=32, num_workers=4, output_dir="evaluation_results"):
    """
    Comprehensive evaluation of a trained model with class-specific metrics.
    
    Args:
        model_path (str): Path to the saved PyTorch model
        val_dataset_path (str): Path to the validation dataset
        batch_size (int): Batch size for evaluation
        num_workers (int): Number of workers for data loading
        output_dir (str): Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and dataset
    device = get_device()
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    
    _,_,val,species_labels, _ = manifest_generator_wrapper(dominant_threshold, export=True)
    val_dataset = CustomDataset(val, train=False, img_size=(160, 160))
    # val_dataset = torch.load(val_dataset_path, weights_only=False, map_location=device)
    
    # Get class names
    # Try different approaches to get class names
    try:
        # NEW: Check if this is a BinaryInsectDataset
        if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'dominant_class_idx'):
            # This is our BinaryInsectDataset
            binary_dataset = val_dataset.dataset
            dominant_class_name = binary_dataset.dataset.classes[binary_dataset.dominant_class_idx]
            class_names = [dominant_class_name, "Other Species"]
            print(f"Detected binary classification dataset:")
            print(f"  Class 0: {class_names[0]} (dominant species)")
            print(f"  Class 1: {class_names[1]} (all others)")
        elif isinstance(val_dataset, BinaryInsectDataset):
            # Direct BinaryInsectDataset instance
            dominant_class_name = val_dataset.dataset.classes[val_dataset.dominant_class_idx]
            class_names = [dominant_class_name, "Other Species"]
            print(f"Detected binary classification dataset:")
            print(f"  Class 0: {class_names[0]} (dominant species)")
            print(f"  Class 1: {class_names[1]} (all others)")
        else:
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
                        # Fallback: Check model output size to determine if binary
                        # Load model temporarily to check output size
                        temp_model = torch.load(model_path, weights_only=False, map_location='cpu')
                        if hasattr(temp_model, 'classifier') and hasattr(temp_model.classifier, 'linear'):
                            num_model_classes = temp_model.classifier.linear.out_features
                            if num_model_classes == 2:
                                class_names = ["Dominant Species", "Other Species"]
                                print("Detected binary classification from model output size")
                            else:
                                class_names = [f"Class {i}" for i in range(num_model_classes)]
                        else:
                            # Ultimate fallback
                            class_names = ["Class 0", "Class 1"]
                            
    except Exception as e:
        print(f"Warning: Could not extract class names automatically: {e}")
        print("Checking model for binary classification...")
        
        # Check model output size as fallback
        try:
            temp_model = torch.load(model_path, weights_only=False, map_location='cpu')
            if hasattr(temp_model, 'classifier') and hasattr(temp_model.classifier, 'linear'):
                num_model_classes = temp_model.classifier.linear.out_features
                if num_model_classes == 2:
                    class_names = ["Dominant Species", "Other Species"]
                    print("Using binary classification based on model output size")
                else:
                    class_names = [f"Class {i}" for i in range(num_model_classes)]
            else:
                class_names = ["Class 0", "Class 1"]  # Default binary
        except:
            class_names = ["Class 0", "Class 1"]  # Safe fallback
    
    num_classes = len(class_names)
    
    print(f"Loaded model from {model_path}")
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
    
    # Evaluate model
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            # Get predicted class
            _, preds = torch.max(outputs, 1)
            
            # Convert to probabilities with softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    label_thresholds = 3
    
    #If the model labels is greater than the label threshold, we will count that as a dominant species
    if num_classes > label_thresholds:
        all_preds = np.where(all_probs.max(axis=1) > dominant_threshold, all_preds, num_classes - 1)
        all_labels = np.where(all_labels > dominant_threshold, all_labels, num_classes - 1)
    
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
        labels=range(num_classes),
        target_names=class_names,
        output_dict=True
    )
       
    print(report)
    
    # Convert classification report to DataFrame for easier handling\
    report_df = pd.DataFrame(report).transpose()
    
    #calculate precision and recall for the "Other Species" class
    pred, rec, f1 = calculate_other_precision_recall_f1(all_labels, all_preds, num_classes)
    report_df.loc['Other Species', 'precision'] = pred
    report_df.loc['Other Species', 'recall'] = rec
    report_df.loc['Other Species', 'f1-score'] = f1

    # Save classification report
    report_df.to_csv(f"{output_dir}/classification_report.csv")
    
    # Print class-specific metrics
    print("\n===== CLASS-SPECIFIC METRICS =====")
    
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
            conf_scores = all_probs[pred_class_i, i]
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
    plt.tight_layout()
    plt.title('Confusion Matrix')
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
    
    print(f"\nEvaluation complete. Results saved to {output_dir}/")
    
    return {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'class_metrics': class_df,
        'confusion_matrix': cm_df
    }

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Evaluate trained model with detailed metrics')
    # parser.add_argument('--model', type=str, default="models/mcunet_haute_garonne_8_species.pth", help='Path to model file (.pth)')
    # parser.add_argument('--dataset', type=str, default="val_dataset.pt", help='Path to validation dataset')
    # parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    # parser.add_argument('--output', type=str, default="evaluation_results", help='Output directory for results')
    
    # args = parser.parse_args()
    model_name = [3,4,5,6,7,8,10,14]
    dominant_threshold = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    for model, threshold in zip(model_name, dominant_threshold):
        evaluate_model(
            model_path=f"models/mcunet-in2_haute_garonne_{model}_species.pth",
            dominant_threshold=threshold,
            batch_size=32,
            num_workers=4,
            output_dir=f"evaluation_results_{model}_{threshold:.2f}"  # Use model number and threshold in output dir name
        )

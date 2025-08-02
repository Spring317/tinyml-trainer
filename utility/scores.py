from typing import List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def calculate_scores(all_labels: np.ndarray, all_preds: np.ndarray) -> tuple:
    """
    Calculate various classification metrics.

    Parameters:
    all_labels (array-like): True labels.
    all_preds (array-like): Predicted labels.

    Returns:
    tuple: A tuple containing accuracy, precision, recall, F1 score.
    """
    try:
        overall_accuracy = accuracy_score(all_labels, all_preds)
        overall_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        overall_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        overall_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return overall_accuracy, overall_precision, overall_recall, overall_f1
    except Exception as e:
        print(f"Error calculating scores: {e}")
        return 0.0, 0.0, 0.0, 0.0


def get_classification_report(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int,
    class_names: List,
    output_dict: bool = True,
) -> dict:
    """
    Generate a classification report with robust error handling.

    Parameters:
    all_labels (np.ndarray): True labels.
    all_preds (np.ndarray): Predicted labels.
    num_classes (int): Number of unique labels.
    class_names (List): List of target names for the classes.
    output_dict (bool): If True, returns the report as a dictionary.

    Returns:
    dict: Classification report as a dictionary.
    """
    try:
        # Ensure class_names is a list of strings
        if not isinstance(class_names, list):
            class_names = list(class_names)
        
        # Convert all class names to strings
        class_names = [str(name) for name in class_names]
        
        # Ensure we have the right number of class names
        if len(class_names) != num_classes:
            print(f"Warning: class_names length ({len(class_names)}) != num_classes ({num_classes})")
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        # Get unique labels to determine the actual classes present
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        actual_labels = [i for i in range(num_classes) if i in unique_labels]
        
        # Filter class names for actual labels
        filtered_class_names = [class_names[i] if i < len(class_names) else f"Class_{i}" 
                               for i in actual_labels]
        
        print(f"Generating report for {len(actual_labels)} classes: {filtered_class_names[:5]}...")
        
        report = classification_report(
            all_labels,
            all_preds,
            labels=actual_labels,
            target_names=filtered_class_names,
            output_dict=output_dict,
            zero_division=0
        )
        return report
        
    except Exception as e:
        print(f"Error generating classification report: {e}")
        return {}


def calculate_other_precision_recall_f1(y_true, y_pred, num_classes):
    try:
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
        mod_recall = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        mod_f1 = (
            2 * (mod_precision * mod_recall) / (mod_precision + mod_recall)
            if (mod_precision + mod_recall) > 0
            else 0.0
        )
        return mod_precision, mod_recall, mod_f1
        
    except Exception as e:
        print(f"Error calculating other class metrics: {e}")
        return 0.0, 0.0, 0.0

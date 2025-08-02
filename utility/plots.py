import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix_heatmap(cm: list, class_names: list, output_dir: str):
    """Plot a confusion matrix heatmap.
    Args:
        cm (list): Confusion matrix as a 2D list.
        class_names (list): List of class names corresponding to the confusion matrix.
        output_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()


def plot_class_f1_scores(class_df: list, output_dir: str):
    """Plot F1 scores for each class.
    Args:
        class_df (list): DataFrame containing class names and their corresponding F1 scores.
        output_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x="class_name", y="f1_score", data=class_df)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_f1_scores.png", dpi=300)


def plot_class_precision_recall(class_df: list, output_dir: str):
    """Plot precision and recall for each class.
    Args:
        class_df (list): DataFrame containing class names, precision, and recall.
        output_dir (str): Directory where the plot will be saved.
    """

    plt.figure(figsize=(14, 6))
    class_df_melted = pd.melt(
        class_df,  # type: ignore
        # Convert the DataFrame to a long format for easier plotting
        id_vars=["class_name"],
        value_vars=["precision", "recall"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(x="class_name", y="score", hue="metric", data=class_df_melted)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision and Recall by Class")
    plt.xticks(rotation=90)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall.png", dpi=300)


# def plot_class_confidence(class_confidence: list, output_dir: str):
#     """Plot confidence scores for each class.
#     Args:
#         class_confidence: DataFrame containing class names and their corresponding confidence scores.
#         output_dir (str): Directory where the plot will be saved.
#     """
#
#     if len(class_confidence) > 0:
#         plt.figure(figsize=(14, 6))
#         ax = sns.barplot(x="class_name", y="mean_confidence", data=confidence_df)
#         plt.errorbar(
#             x=np.arange(len(confidence_df)),
#             y=confidence_df["mean_confidence"],
#             yerr=confidence_df["std_confidence"],
#             fmt="none",
#             capsize=5,
#             color="black",
#         )
#         plt.xlabel("Class")
#         plt.ylabel("Mean Confidence Score")
#         plt.title("Mean Prediction Confidence by Class")
#         plt.xticks(rotation=90)
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/class_confidence.png", dpi=300)

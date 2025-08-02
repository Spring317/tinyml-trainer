from data_prep import data_loader

# from pipe_model import PipeModel
from utility.scores import (
    get_classification_report,
    calculate_scores,
    calculate_other_precision_recall_f1,
)
from torch.utils.data import DataLoader
from utility.utilities import manifest_generator_wrapper, get_support_list
from pipeline.dataset_loader import CustomDataset
from utility.utilities import get_device
from data_prep.data_loader import DataLoaderCreator
from data_prep.class_handler import get_class_info_for_evaluation
from models.model_handler import ModelHandler
import argparse
import pandas as pd
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


def evaluate_ensemble(
    model_paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    dominant_threshold: float = 0.2,
    start_rank: int = 0,
    full_dataset: bool = False,
    output_dir: str = "output"
):
    """
    Evaluate an ensemble of models on a dataset.

    Args:
        model_paths (List[str]): Paths to the model files.
        batch_size (int): Size of each batch for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
        dominant_threshold (float): Threshold for dominant species.
        start_rank (int): Starting rank for dataset partitioning.
        full_dataset (bool): Whether to use the full dataset or not.

    Returns:
        Dict[str, float]: Evaluation scores including accuracy, precision, recall, and F1 score.
    """

    device = get_device()
    _, _, val_images, species_labels, species_composition = manifest_generator_wrapper(1.0)

    # with open("./data/haute_garonne/dataset_species_labels_full_bird_insect.json") as file:
    #     species_labels = json.load(file)

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
    # # Fix: Get class info properly
    # try:
    #     # Try the full function signature
    #     class_names, num_classes, class_to_idx = get_class_info_for_evaluation(
    #         start_rank=start_rank,
    #         number_of_dominant_classes=None,
    #         model_path=model_paths[1] if len(model_paths) > 1 else model_paths[0]
    #     )
        
    #     # Ensure class_names is a list of strings
    #     if isinstance(class_names, list) and len(class_names) > 0:
    #         if isinstance(class_names[0], list):
    #             # If it's a list of lists, flatten it
    #             class_names = [str(item) for sublist in class_names for item in sublist]
    #         else:
    #             # Ensure all items are strings
    #             class_names = [str(name) for name in class_names]
    #     else:
    #         # Fallback to generic names
    #         class_names = [f"Species_{i}" for i in range(144)]  # Adjust as needed
    #         num_classes = len(class_names)
            
    # except Exception as e:
    #     print(f"Warning: Could not get class info properly: {e}")
    #     print("Using fallback class names...")
        
    #     # Fallback: Create generic class names
    #     if full_dataset:
    #         num_classes = 144  # Or whatever your full dataset has
    #     else:
    #         num_classes = 3   # For 3-class setup
            
    #     class_names = [f"Species_{i}" for i in range(num_classes)]
    
    # print(f"📋 Using {num_classes} classes")
    # print(f"   Sample class names: {class_names[:5]}...")
    
    # data_loader_creator = DataLoaderCreator(
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     full_dataset=full_dataset,
    #     start_rank=start_rank,
    # )


    # Load the dataset
    # _, val_loader, _, actual_num_classes = data_loader_creator.create_dataloader()
    # if val_loader is None:
    #     raise ValueError(
    #         "DataLoader could not be created. Check your dataset path and parameters."
    #     )
    
    # # Update num_classes based on actual dataloader
    # if actual_num_classes is not None:
    #     num_classes = actual_num_classes
    #     # Adjust class_names if needed
    #     if len(class_names) != num_classes:
    #         class_names = [f"Species_{i}" for i in range(num_classes)]
    
    # print(f"✅ Dataset loaded with {num_classes} classes")
    
    # Load models
    model1 = torch.load(model_paths[0], map_location=device, weights_only=False)
    model2 = torch.load(model_paths[1], map_location=device, weights_only=False)

    model1.eval()
    model2.eval()

    print("🔄 Starting evaluation...")
    
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        # Tracking variables
        model2_used_count = 0
        
        # Detailed tracking
       
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating models")):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get Model 1 predictions
            outputs1 = model1(inputs)
            probs1 = torch.softmax(outputs1, dim=1)
            preds = torch.argmax(outputs1, dim=1)

            if preds.cpu().numpy().item() == 2:
                output2s = model2(inputs)
                preds = torch.argmax(output2s, dim=1)
                model2_used_count += 1
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    print(f"✅ Evaluation complete. Model 2 used {model2_used_count} times out of {len(val_loader.dataset)} samples."
          )
    accuracy, precision, recall, f1 = calculate_scores(
        np.concatenate(all_labels),
        np.concatenate(all_preds), 
        )
    classification_report = get_classification_report(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        class_names=species_names,
        num_classes=len(species_names),
        output_dict=True,
    )
    #save classification report
    report_df = pd.DataFrame(classification_report).transpose() 
    report_df.to_csv(f"{output_dir}/classification_report.csv", index=True)

    return accuracy, precision, recall, f1, classification_report
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of models.")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the model files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker threads for DataLoader.",
    )
    parser.add_argument(
        "--dominant_threshold",
        type=float,
        default=0.2,
        help="Threshold for dominant species.",
    )
    parser.add_argument(
        "--start_rank",
        type=int,
        default=0,
        help="Starting rank for dataset partitioning.",
    )
    parser.add_argument(
        "--full_dataset",
        action="store_true",
        help="Use the full dataset for evaluation.",
    )

    args = parser.parse_args()

    scores = evaluate_ensemble(
        args.model_paths,
        args.batch_size,
        args.num_workers,
        args.dominant_threshold,
        args.start_rank,
        args.full_dataset,
    )

    print("Evaluation Scores:", scores)

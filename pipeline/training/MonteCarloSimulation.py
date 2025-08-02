import random
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import scipy.special
from tqdm import tqdm
from typing import Dict, List, Union, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from dataset_builder.core.utility import load_manifest_parquet
from pipeline.utility import generate_report, get_support_list, preprocess_eval_opencv

class MonteCarloSimulation:
    """
    A class for running Monte Carlo simulations to evaluate the performance and robustness of an ONNX image classification model.

    The simulation randomly samples species and images, runs model inference, and collects evaluation statistics such as communication rates, false positive rates, and accuracy scores.
    It also supports saving detailed reports and plotting confusion matrices.

    Args:
        model_path (str): 
            Path to the ONNX model to load.
        data_manifest (Union[str, List[Tuple[str, int]]]): 
            Either the path to a Parquet file containing (image_path, label) pairs or a preloaded list of (image_path, label) samples.
        dataset_species_labels (Dict[int, str]): 
            Mapping from integer class IDs to species names.
        input_size (Tuple[int, int], optional): 
            Expected (height, width) for input images. Defaults to (224, 224).
        providers (List[str], optional): 
            List of ONNXRuntime providers for inference (e.g., CPU, CUDA). Defaults to ["CPUExecutionProvider"].

    Notes:
        - The simulation first ensures one sample per species, then fills the rest of the batch using weighted random sampling based on species image availability.
        - The confusion matrix is saved to a file if requested.
        - If `save_path` is provided, a detailed CSV report of classification metrics is generated.
    """
    def __init__(
        self, 
        model_path, 
        data_manifest: Union[str, List[Tuple[str, int]]], 
        dataset_species_labels, 
        is_inception_v3: bool,
        input_size=(224, 224),
        providers: List[str]=["CPUExecutionProvider"]
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.species_labels: Dict[int, str] = dataset_species_labels
        
        self.other_class_id = int(self._get_other_id())
        if isinstance(data_manifest, str):
            self.data_manifest = load_manifest_parquet(data_manifest)
        else:
            self.data_manifest = data_manifest
        self.species_to_images = defaultdict(list)
        self.species_probs = {}
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        for image_path, species_id in self.data_manifest:
            self.species_to_images[species_id].append(image_path)

        total_images = sum(len(imgs) for imgs in self.species_to_images.values())
        self.species_probs = {
            int(species_id): len(images) / total_images
            for species_id, images in self.species_to_images.items()
        }
        self.is_inception_v3 = is_inception_v3


    def _get_other_id(self):
        """
        Retrieves the class ID corresponding to the "Other" species label.

        This method inverts the species_labels dictionary (mapping from class ID -> label) to label -> class ID, and attempts to find the class ID associated with "Other".
        If "Other" is not present, returns -1 as a default.

        Returns:
            int: The class ID of the "Other" species if found, otherwise -1.
        """
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.species_labels.items())
        return species_labels_flip.get("Other", -1)


    def _infer_one(self, image_path: str) -> Optional[Tuple[int, float]]:
        """
        Performs model inference on a single input image and returns the top-1 predicted class index along with its probability score.

        Args:
            image_path (str): 
                Path to the input image file.

        Returns:
            Optional[Tuple[int, float]]: 
                A tuple containing:
                    - The predicted class index (int).
                    - The associated top-1 probability score (float).
                Returns None if an error occurs during preprocessing or inference.

        Notes:
            - Images are preprocessed using `preprocess_eval_opencv` to match the model's input size.
            - Softmax is applied to model outputs to obtain probability distributions.
        """
        try:
            img = preprocess_eval_opencv(image_path, *self.input_size, is_inception_v3=self.is_inception_v3)
            outputs = self.session.run(None, {self.input_name: img})
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None


    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Plots and saves a confusion matrix for the model's predictions on the given true labels.

        Args:
            y_true (List[int]): 
                Ground truth class IDs.
            y_pred (List[int]): 
                Predicted class IDs from the model.

        Notes:
            - Uses `sklearn.metrics.ConfusionMatrixDisplay` to visualize the matrix.
            - Axis labels are derived from `self.species_labels` (class ID to name).
            - The plot is saved as a PNG file with the filename:
                "MonteCarloConfusionMatrix_<model_name>.png"
            - The figure size is large (40x40 inches).
        """
        cm = confusion_matrix(y_true, y_pred, labels=list(map(int, self.species_labels.keys())))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.species_labels.values()))
        fig, ax = plt.subplots(figsize=(40, 40))
        disp.plot(ax=ax, xticks_rotation=50, cmap="Blues", colorbar=True)
        plt.title("Confusion Matrix (Monte Carlo Simulation)")
        plt.tight_layout()
        plt.savefig(f"MonteCarloConfusionMatrix_{os.path.basename(self.model_path).replace('.onnx', '')}.png")


    def _false_positive_rate(self, y_true, y_pred):
        """
        Computes the false positive rate (FPR) for the "Other" class predictions.

        Specifically, it measures how often the model incorrectly predicts a sample that should be classified as "Other" (communication) into a different (local) class.

        Args:
            y_true (List[int]): 
                Ground truth class IDs.
            y_pred (List[int]): 
                Predicted class IDs from the model.

        Returns:
            float: 
                The false positive rate, calculated as:
                    FPR = False Positives / (False Positives + True Negatives)
                Returns 0.0 if there are no samples of the "Other" class.

        Notes:
            - "False Positive" (FP): 
                True label is "Other", but the model predicts a different class.
            - "True Negative" (TN): 
                True label is "Other", and the model correctly predicts "Other".
            - If there are no "Other" class samples, the function returns 0.0.
        """
        # true: local / false: communication
        # fp: it should be communicate but model predict as local
        # tn: it should be communicate, model predict as communicate
        fp_count = 0
        tn_count = 0
        for label, predict in zip(y_true, y_pred):
            if label == self.other_class_id:
                if predict != label:
                    fp_count += 1
                else:
                    tn_count += 1
        if (fp_count + tn_count) == 0:
            return 0.0
        return fp_count / (fp_count + tn_count) 

    def run_simulation(
        self,
        species_labels: Dict[int, str],
        species_composition: Dict[str, int],
        num_runs: int=30,
        sample_size: int=1000,
        plot_confusion_matrix: bool=False,
        save_path=None,
    ):
        """
        Runs a Monte Carlo simulation to evaluate model performance by repeatedly sampling 
        and classifying random images across multiple runs.

        For each run, a balanced sample containing at least one image per species is generated. 
        The model's predictions are collected, and key metrics such as communication rate, 
        false positive rate (FPR), and overall classification accuracy are computed.

        Args:
            species_labels (Dict[int, str]): 
                Mapping from species ID to species name.
            species_composition (Dict[str, int]): 
                Dictionary mapping species names to the number of available images.
            num_runs (int, optional): 
                Number of independent simulation runs to perform. Defaults to 30.
            sample_size (int, optional): 
                Total number of images to sample per run. Defaults to 1000.
            plot_confusion_matrix (bool, optional): 
                Whether to generate and save a confusion matrix after simulation. Defaults to False.
            save_path (str, optional): 
                Directory to save a CSV report summarizing classification performance. 
                If None, no report is saved.

        Notes:
            - Each run guarantees at least one sample per species.
            - The remaining images are sampled based on species sampling probabilities.
            - Communication rate is defined as the proportion of predictions labeled as "Other."
            - False positive rate (FPR) is calculated specifically for the "Other" class.
            - If `save_path` is provided, a detailed classification report CSV is saved.
            - If `plot_confusion_matrix` is True, a confusion matrix PNG is generated and saved.

        Output Files (optional):
            - `MonteCarloConfusionMatrix_<model_name>.png`: 
                Saved confusion matrix plot (if enabled).
            - `<model_name>.csv`: 
                Saved classification report containing per-class metrics (if enabled).
        """
        species_names = list(species_labels.values())
        total_support_list = get_support_list(species_composition, species_names)
        comm_rates: List[float] = []
        all_true: List[int] = []
        all_pred: List[int] = []

        for run in range(num_runs):
            y_true: List[int] = []
            y_pred: List[int] = []
            sampled_species = []
            for species in self.species_labels.keys():
                sampled_species.append(species)

            remaining_k = sample_size - len(sampled_species)
            sampled_species += random.choices(
                population=list(self.species_labels.keys()),
                weights=[self.species_probs[int(sid)] for sid in self.species_labels.keys()],
                k=remaining_k
            )
            random.shuffle(sampled_species)

            num_comm = 0
            num_local = 0
            correct = 0

            for species_id in tqdm(sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False):
                image_list = self.species_to_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)
                result = self._infer_one(image_path)
                if result is None:
                    continue
                y_true.append(int(species_id))
                y_pred.append(int(result[0]))
                if result[0] == int(species_id):
                    correct += 1
                top1_idx, top1_prop = result
                if top1_idx == self.other_class_id:
                    num_comm += 1
                else:
                    num_local += 1
            
            total_pred = num_comm + num_local
            comm_rate = num_comm / total_pred if total_pred else 0
            comm_rates.append(comm_rate)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        model_name = os.path.basename(self.model_path).replace(".onnx", "")

        if save_path:
            accuracy = accuracy_score(all_true, all_pred)
            df = generate_report(all_true, all_pred, species_names, total_support_list, float(accuracy)) 

            os.makedirs(save_path, exist_ok=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                df.to_csv(os.path.join(save_path, f"{model_name}.csv"))

        print(f"Average communication for {model_name}: {sum(comm_rates)/len(comm_rates)}", end=" ")
        print(f"FPR: {self._false_positive_rate(all_true, all_pred)}")
        if plot_confusion_matrix:
            self._plot_confusion_matrix(all_true, all_pred)

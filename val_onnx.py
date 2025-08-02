import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np

from CustomDataset import CustomDataset  # Your existing dataset class
from utilities import manifest_generator_wrapper, preprocess_eval_opencv
from typing import List, Tuple
import scipy


def validate_onnx_model(
    onnx_path: str,
    dataset: List[Tuple[str, int]],
    verbose: bool = True,
):
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    y_true, y_pred = [], []

    for image, label in tqdm(dataset, desc="Validating ONNX"):
        image = preprocess_eval_opencv(image, 160, 160)

        # Run ONNX inference
        outputs = session.run([output_name], {input_name: image})  # shape: (B, C)
        preds = np.argmax(outputs[0], axis=1)
        probabilities = scipy.special.softmax(outputs[0], axis=1)
        top1_idx = int(np.argmax(probabilities[0]))
        top1_prob = float(probabilities[0][top1_idx])
        print(top1_prob)

        y_true.append(label)
        y_pred.append(preds)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    if verbose:
        print(f"ONNX Accuracy: {acc:.4f}")
        print(f"ONNX Macro F1-score: {macro_f1:.4f}")

    return acc, macro_f1


all_data, _, _, _, _ = manifest_generator_wrapper(0.3, export=True)

validate_onnx_model("models/mcunet_haute_garonne_8_species_infer_q8.onnx", all_data)

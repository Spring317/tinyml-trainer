import torch
from tqdm import tqdm
from typing import Tuple, List
from sklearn.metrics import f1_score
import os


class ModelHandler:
    def __init__(self, device):
        """
        Initializes the Model_Handler with the specified model path.

        :param model_path: Path to the model file.
        """
        self.device = device

    def eval_one_epoch(
        self, model, dataloader
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Evaluates the model for one epoch using the provided dataloader.

        :param model: The model to evaluate.
        :param dataloader: The dataloader containing the data to evaluate on.
        :return: The evaluation results.
        """
        with torch.no_grad():
            model.eval()
            preds, labels, probs = [], [], []
            loop = tqdm(dataloader, desc="Evaluating", unit="batch", leave=False)
            for images, label in loop:
                images, label = images.to(self.device), label.to(self.device)
                outputs = model(images)

                # Get predicted class
                _, pred = torch.max(outputs, 1)

                # Convert to probabilities with softmax
                prob = torch.nn.functional.softmax(outputs, dim=1)
                pred = pred.cpu()
                label = label.cpu()
                prob = prob.cpu()

                preds.extend(pred.numpy())
                labels.extend(label.numpy())
                probs.extend(prob.numpy())

        return preds, labels, probs

    def train_one_epoch(
        self, model, dataloader, criterion, optimizer
    ) -> Tuple[float, float]:
        """
            Trains the given model for one epoch on the provided DataLoader.

            This function performs standard supervised learning with forward pass,
            loss computation, backpropagation, and optimizer updates. It also includes
            a label sanity check to ensure labels fall within valid class index range.
        CLASS_NAME
            Args:
                model (torch.nn.Module):
                    The model to be trained.
                dataloader (DataLoader[CustomDataset]):
                    A DataLoader that yields batches of (image, label) pairs.
                criterion:
                    A loss function (e.g., nn.CrossEntropyLoss).
                optimizer:
                    A PyTorch optimizer (e.g., torch.optim.Adam or SGD).
                device (torch.device):
                    The device on which to perform computation (CPU or GPU).

            Returns:
                Tuple[float, float]:
                    A tuple containing:
                        - The average training loss over the entire dataset.
                        - The training accuracy over the entire dataset.

            Notes:
                - A tensor guard checks that the ground-truth labels fall within the valid range `[0, num_classes - 1]` based on model output shape.
                - Accumulates total correct predictions and total loss for final reporting.
        """
        model.train()
        total_loss, correct = 0.0, 0

        loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)

        checked_labels = False

        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)

            # tensor guard
            if not checked_labels:
                num_classes = model(images).shape[1]
                label_min = labels.min().item()
                label_max = labels.max().item()

                if labels.min() < 0 or labels.max() >= num_classes:
                    raise ValueError(
                        f"Invalid labels detected!\n"
                        f"Labels: {labels}\n"
                        f"Min: {label_min}, Max: {label_max}\n"
                        f"Model output classes: {num_classes}"
                    )
                checked_labels = True

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            loop.set_postfix(loss=f"{loss.detach().item():.3f}")

        avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
        accuracy = correct / len(dataloader.dataset)  # type: ignore
        return avg_loss, accuracy

    def train_validate(
        self, model, dataloader, criterion
    ) -> Tuple[float, float, float]:
        """
        Trains the model for one epoch with sparsity regularization applied via a pruner.

        This function is used during a warm-up phase of sparse training, where a regularizer (e.g., L1/L2 penalty on weights or activations) is applied to encourage sparsity before actual pruning is performed.

        Args:
            model (torch.nn.Module):
                The model to be trained.
            dataloader (DataLoader[CustomDataset]):
                A DataLoader that yields batches of (image, label) pairs.
            criterion:
                The loss function used to train the model (e.g., CrossEntropyLoss).
            pruner:
                A sparsity regularizer object that provides:
                    - `update_regularizer()`: Called once before training.
                    - `regularize(model)`: Called on each backward pass to apply regularization.
            device (torch.device):
                The device on which to perform training (CPU or GPU).

        Returns:
            Tuple[float, float]:
                A tuple containing:
                    - Average loss over the epoch.
                    - Accuracy over the entire dataset.
        """

        model.eval()
        total_loss, correct = 0.0, 0
        loop = tqdm(dataloader, desc="Validating", unit="batch", leave=False)
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                total_loss += loss.detach().item() * images.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
        accuracy = correct / len(dataloader.dataset)  # type: ignore
        macro_f1 = f1_score(true_labels, pred_labels, average="macro")

        return avg_loss, accuracy, float(macro_f1)

    def save_model(
        self,
        model,
        name: str,
        save_path: str,
        img_size: Tuple[int, int],
    ):
        """
        Saves a PyTorch model in both `.pth` and ONNX formats.

        This function exports the given model to:
            - PyTorch format (.pth) using `torch.save()`
            - ONNX format (.onnx) using `torch.onnx.export()`, with support for dynamic batch sizes

        Args:
            model (torch.nn.Module):
                The trained PyTorch model to be saved.
            name (str):
                Base name for the output files (e.g., 'mobilenetv3').
            save_path (str):
                Directory where the model files will be saved. Will be created if it doesn't exist.
            device (torch.device):
                Device on which to create the dummy input tensor for ONNX export.
            img_size (Tuple[int, int]):
                Expected input image size as (height, width) for dummy input.

        Output Files:
            - `<save_path>/<name>.pth`: PyTorch serialized model.
            - `<save_path>/<name>.onnx`: ONNX exported model with dynamic batch dimension.

        Notes:
            - ONNX export uses `opset_version=14` and includes constant folding for optimization.
            - Assumes the model expects input shape `(N, 3, H, W)` where H and W are from `img_size`.
            - Dynamic axes allow for variable batch sizes during ONNX inference.
        """
        os.makedirs(save_path, exist_ok=True)
        pytorch_path = os.path.join(save_path, f"{name}.pth")
        torch.save(model, pytorch_path)
        print(f"Saved Pytorch model to {pytorch_path}")

        dummy_input = torch.randn(1, 3, *img_size, device=self.device)
        onnx_path = os.path.join(save_path, f"{name}.onnx")
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Exported ONNX model to {onnx_path}")

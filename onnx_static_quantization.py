import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
from datetime import datetime

class ONNXCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for ONNX quantization
    """
    def __init__(self, calibration_data, input_name='input'):
        self.calibration_data = calibration_data
        self.input_name = input_name
        self.current_index = 0
        
    def get_next(self):
        if self.current_index >= len(self.calibration_data):
            return None
        
        # Get next batch
        batch = self.calibration_data[self.current_index]
        self.current_index += 1
        
        # Convert to numpy if it's a torch tensor
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()
        
        # Return as dictionary with input name as key
        return {self.input_name: batch}
    
    def rewind(self):
        self.current_index = 0

def create_calibration_data_for_onnx(config, batch_size=1):
    """
    Create calibration data specifically for ONNX quantization
    """
    # Extract dataset configuration
    dataset_name = getattr(config, 'dataset', 'cifar10')
    data_path = getattr(config, 'data_path', './data')
    calibration_samples = getattr(config, 'calibration_samples', 100)  # Smaller for ONNX
    
    print(f"Creating calibration data for {dataset_name}...")
    
    # Define transforms (no augmentation for calibration)
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        calibration_dataset = torchvision.datasets.CIFAR10(
            root=data_path, 
            train=False, 
            download=True, 
            transform=transform
        )
        
    elif dataset_name.lower() == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        calibration_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, 'val'),
            transform=transform
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create subset for calibration
    np.random.seed(42)  # For reproducibility
    if len(calibration_dataset) > calibration_samples:
        indices = np.random.choice(len(calibration_dataset), calibration_samples, replace=False)
        calibration_dataset = Subset(calibration_dataset, indices)
    
    # Create data loader with batch_size=1 for ONNX (typical for calibration)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    # Collect calibration data
    calibration_data = []
    print("Collecting calibration data...")
    
    for batch_idx, (data, _) in enumerate(calibration_loader):
        # Convert to numpy and add to calibration data
        calibration_data.append(data.cpu().numpy())
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(calibration_loader)} batches")
    
    print(f"Created {len(calibration_data)} calibration samples")
    return calibration_data

def quantize_onnx_model(model_path, calibration_data, output_path, input_name='input'):
    """
    Perform static quantization on ONNX model
    """
    print(f"Starting ONNX static quantization...")
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")
    
    # Create calibration data reader
    calibration_data_reader = ONNXCalibrationDataReader(calibration_data, input_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Quantization configuration
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QOperator,  # Use QOperator format
        op_types_to_quantize=['Conv', 'MatMul', 'Add', 'Mul'],  # Common ops to quantize
        per_channel=False,  # Per-tensor quantization (more compatible)
        reduce_range=False,  # Don't reduce range for better accuracy
        activation_type=QuantType.QUInt8,  # 8-bit unsigned int for activations
        weight_type=QuantType.QInt8,  # 8-bit signed int for weights
        optimize_model=True,  # Optimize the model
        use_external_data_format=False,  # Keep model in single file
        calibrate_method='MinMax'  # Calibration method
    )
    
    print(f"Quantization completed successfully!")
    return output_path

def validate_quantized_model(original_model_path, quantized_model_path, test_data_sample):
    """
    Validate the quantized model by comparing outputs
    """
    print("Validating quantized model...")
    
    # Load original model
    original_session = onnxruntime.InferenceSession(original_model_path)
    original_input_name = original_session.get_inputs()[0].name
    
    # Load quantized model
    quantized_session = onnxruntime.InferenceSession(quantized_model_path)
    quantized_input_name = quantized_session.get_inputs()[0].name
    
    # Run inference on both models
    test_input = test_data_sample[0] if isinstance(test_data_sample, list) else test_data_sample
    
    original_output = original_session.run(None, {original_input_name: test_input})
    quantized_output = quantized_session.run(None, {quantized_input_name: test_input})
    
    # Compare outputs
    original_pred = original_output[0]
    quantized_pred = quantized_output[0]
    
    # Calculate metrics
    mse = np.mean((original_pred - quantized_pred) ** 2)
    mae = np.mean(np.abs(original_pred - quantized_pred))
    max_diff = np.max(np.abs(original_pred - quantized_pred))
    
    print(f"Validation Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max Difference: {max_diff:.6f}")
    
    # Check if outputs are close (within reasonable tolerance)
    if mae < 0.1:  # Adjust threshold as needed
        print("✓ Quantized model validation passed!")
    else:
        print("⚠ Quantized model shows significant differences from original")
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'max_diff': float(max_diff)
    }

def get_model_size(model_path):
    """
    Get model file size in MB
    """
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)

def main():
    # Configuration (similar to train.py)
    class QuantizationConfig:
        dataset = 'cifar10'  # or 'imagenet'
        data_path = './data'
        calibration_samples = 100  # Smaller number for ONNX calibration
        batch_size = 1  # Typically 1 for ONNX calibration
    
    config = QuantizationConfig()
    
    # Model paths
    original_model_path = './model/mcunet_haute_garonne_other_20.onnx'
    quantized_model_path = './quantized_models/mcunet_haute_garonne_other_20_quantized.onnx'
    
    # Check if original model exists
    if not os.path.exists(original_model_path):
        print(f"Error: Original model not found at {original_model_path}")
        return
    
    # Load original model to get input information
    original_model = onnx.load(original_model_path)
    input_name = original_model.graph.input[0].name
    print(f"Model input name: {input_name}")
    
    try:
        # Create calibration data
        calibration_data = create_calibration_data_for_onnx(config, batch_size=config.batch_size)
        
        # Perform quantization
        quantized_model_path = quantize_onnx_model(
            original_model_path, 
            calibration_data, 
            quantized_model_path,
            input_name
        )
        
        # Validate quantized model
        validation_results = validate_quantized_model(
            original_model_path, 
            quantized_model_path, 
            calibration_data[:1]
        )
        
        # Compare model sizes
        original_size = get_model_size(original_model_path)
        quantized_size = get_model_size(quantized_model_path)
        compression_ratio = original_size / quantized_size
        
        # Save quantization report
        report = {
            'original_model': original_model_path,
            'quantized_model': quantized_model_path,
            'dataset': config.dataset,
            'calibration_samples': config.calibration_samples,
            'original_size_mb': round(original_size, 2),
            'quantized_size_mb': round(quantized_size, 2),
            'compression_ratio': round(compression_ratio, 2),
            'validation_results': validation_results,
            'quantization_timestamp': datetime.now().isoformat()
        }
        
        report_path = './quantized_models/quantization_report.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Quantization Summary ===")
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Quantized model saved to: {quantized_model_path}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
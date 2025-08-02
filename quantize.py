from PIL import Image
import numpy as np
import os
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnx
import torch
import torchvision.transforms as transforms
from CustomDataset import CustomDataset
from utilities import manifest_generator_wrapper, get_device

def get_train_transforms(img_size=(160, 160)):
    """
    Get the same transforms used during training to ensure consistency
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # Converts PIL to tensor and normalizes to [0,1]
        # Add any other transforms that were used during training
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Uncomment if used in training
    ])

def representative_data_gen(image_size=(160, 160), num_samples=1000):
    """
    Generate calibration data with the same preprocessing as training
    """
    _, calib, _, _ ,_  = manifest_generator_wrapper(0.5, export=True)
    
    # Use the same transforms as training
    transform = get_train_transforms(image_size)
    
    count = 0 
    for fname, label in calib:
        try:
            print(f"Processing file: {fname}")
            
            # Load image the same way as in CustomDataset
            img = Image.open(fname).convert("RGB")
            print(f"Image size before transform: {img.size}")
            
            # Apply the same transforms as training
            img_tensor = transform(img)
            
            # Convert to numpy for ONNX Runtime (it expects numpy arrays)
            img_np = img_tensor.numpy()
            
            # Add batch dimension
            img_np = np.expand_dims(img_np, 0)
            
            print(f"Final image shape: {img_np.shape}")
            print(f"Image value range: [{img_np.min():.3f}, {img_np.max():.3f}]")
            
            yield {"input": img_np}
            count += 1 
            if count >= num_samples:
                break
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

def representative_data_gen_with_dataset(image_size=(160, 160), num_samples=300):
    """
    Alternative: Use CustomDataset directly to ensure exact same preprocessing
    """
    _, calib, _, _ ,_  = manifest_generator_wrapper(1.0, export=True)
    
    # Create dataset with same parameters as training
    calib_dataset = CustomDataset(calib, train=False, img_size=image_size)
    
    count = 0
    for i in range(min(len(calib_dataset), num_samples)):
        try:
            img_tensor, label = calib_dataset[i]
            
            # Convert tensor to numpy
            img_np = img_tensor.numpy()
            
            # Add batch dimension
            img_np = np.expand_dims(img_np, 0)
            
            if count == 0:  # Print info for first sample
                print(f"Using CustomDataset preprocessing")
                print(f"Image shape: {img_np.shape}")
                print(f"Image value range: [{img_np.min():.3f}, {img_np.max():.3f}]")
            
            yield {"input": img_np}
            count += 1
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

class ImageDataReader(CalibrationDataReader):
    def __init__(self, data_gen):
        self.data_iter = iter(data_gen)

    def get_next(self): # type: ignore
        return next(self.data_iter, None)

def main():
    # Load model
    onnx_model_path = "models/mcunet_haute_garonne_8_species_infer.onnx"
    quantized_model_path = "models/mcunet_haute_garonne_8_species_quantized.onnx"

    print("Choose calibration data preprocessing method:")
    print("1. Using transforms (manual preprocessing)")
    print("2. Using CustomDataset (exact same as training)")
    
    # Use CustomDataset method for exact same preprocessing as training
    use_dataset = True  # Set to False to use manual transforms
    
    if use_dataset:
        print("Using CustomDataset for calibration data preprocessing...")
        data_reader = ImageDataReader(
            representative_data_gen_with_dataset(image_size=(160, 160), num_samples=300)
        )
    else:
        print("Using manual transforms for calibration data preprocessing...")
        data_reader = ImageDataReader(
            representative_data_gen(image_size=(160, 160), num_samples=300)
        )

    print(f"Quantizing model {onnx_model_path} to {quantized_model_path} using 300 samples for calibration...")
    
    try:
        quantize_static(
            model_input=onnx_model_path,
            model_output=quantized_model_path,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            per_channel=True,
            use_external_data_format=False,
        )
        print(f"✓ Quantization completed successfully!")
        print(f"✓ Quantized model saved to: {quantized_model_path}")
        
        # Verify the quantized model
        try:
            quantized_model = onnx.load(quantized_model_path)
            print(f"✓ Quantized model verification passed")
            print(f"  - Model size: {os.path.getsize(quantized_model_path) / (1024*1024):.2f} MB")
            
            # Print input/output info if available
            if quantized_model.graph.input:
                input_shape = [dim.dim_value for dim in quantized_model.graph.input[0].type.tensor_type.shape.dim]
                print(f"  - Input shape: {input_shape}")
            if quantized_model.graph.output:
                output_shape = [dim.dim_value for dim in quantized_model.graph.output[0].type.tensor_type.shape.dim]
                print(f"  - Output shape: {output_shape}")
                
        except Exception as verify_e:
            print(f"Warning: Could not verify model details: {verify_e}")
        
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        print("Common solutions:")
        print("- Ensure the ONNX model path is correct")
        print("- Check that calibration data generator is working")
        print("- Try with fewer calibration samples")
        print("- Verify ONNX Runtime version compatibility")
        raise

if __name__ == "__main__":
    main()

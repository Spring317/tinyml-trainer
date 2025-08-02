import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import mobilenet_v2
from torch.optim import Adam
from PIL import Image
import time
import shutil
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
NUM_WORKERS = 4
IMG_SIZE = 160  # Smaller size for MCU compatibility

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for binary classification
os.makedirs("binary_dataset/train/abaeis_nicippe", exist_ok=True)
os.makedirs("binary_dataset/train/other", exist_ok=True)
os.makedirs("binary_dataset/val/abaeis_nicippe", exist_ok=True)
os.makedirs("binary_dataset/val/other", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Function to copy files
def copy_files(source_dir, dest_dir, limit=None):
    files = os.listdir(source_dir)
    files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if limit is not None and len(files) > limit:
        files = random.sample(files, limit)
        
    for file in files:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(src_path, dest_path)
    
    return len(files)

# Prepare dataset
print("Preparing binary dataset...")

# Step 1: Count Abaeis nicippe samples
source_dir = "/home/quydx/tinyML/inat_2017/Insecta/Abaeis nicippe"
if not os.path.exists(source_dir):
    raise FileNotFoundError(f"Directory not found: {source_dir}")

all_abaeis_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
num_abaeis = len(all_abaeis_files)
print(f"Found {num_abaeis} samples of Abaeis nicippe")

# Step 2: Determine train/val split (80/20)
num_train_abaeis = int(num_abaeis * 0.8)
num_val_abaeis = num_abaeis - num_train_abaeis

# Step 3: Copy Abaeis nicippe samples
train_abaeis_files = random.sample(all_abaeis_files, num_train_abaeis)
val_abaeis_files = [f for f in all_abaeis_files if f not in train_abaeis_files]

# Copy train samples
for file in train_abaeis_files:
    src_path = os.path.join(source_dir, file)
    dest_path = os.path.join("binary_dataset/train/abaeis_nicippe", file)
    shutil.copy(src_path, dest_path)

# Copy val samples
for file in val_abaeis_files:
    src_path = os.path.join(source_dir, file)
    dest_path = os.path.join("binary_dataset/val/abaeis_nicippe", file)
    shutil.copy(src_path, dest_path)

print(f"Copied {len(train_abaeis_files)} Abaeis nicippe samples to train set")
print(f"Copied {len(val_abaeis_files)} Abaeis nicippe samples to validation set")

# Step 4: Sample "Other" class from other folders
insecta_dir = "/home/quydx/tinyML/inat_2017/Insecta"
other_folders = [d for d in os.listdir(insecta_dir) if os.path.isdir(os.path.join(insecta_dir, d)) and d != "Abaeis nicippe"]

if not other_folders:
    raise ValueError("No other insect folders found")

print(f"Found {len(other_folders)} other insect species")

# Collect files for "Other" class
other_train_files = []
other_val_files = []

# We need to sample approximately the same number as Abaeis nicippe
samples_per_species_train = max(1, num_train_abaeis // len(other_folders))
samples_per_species_val = max(1, num_val_abaeis // len(other_folders))

for folder in other_folders:
    folder_path = os.path.join(insecta_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        continue
    
    # Sample train files
    if len(files) <= samples_per_species_train:
        train_samples = files
    else:
        train_samples = random.sample(files, samples_per_species_train)
    
    other_train_files.extend([(folder, f) for f in train_samples])
    
    # Sample val files (exclude those already in train)
    remaining_files = [f for f in files if f not in train_samples]
    
    if len(remaining_files) <= samples_per_species_val:
        val_samples = remaining_files
    else:
        val_samples = random.sample(remaining_files, samples_per_species_val)
    
    other_val_files.extend([(folder, f) for f in val_samples])

# Limit other samples to match Abaeis nicippe counts
if len(other_train_files) > num_train_abaeis:
    other_train_files = random.sample(other_train_files, num_train_abaeis)
if len(other_val_files) > num_val_abaeis:
    other_val_files = random.sample(other_val_files, num_val_abaeis)

# Copy "Other" train samples
for folder, file in other_train_files:
    src_path = os.path.join(insecta_dir, folder, file)
    dest_path = os.path.join("binary_dataset/train/other", f"{folder}_{file}")
    shutil.copy(src_path, dest_path)

# Copy "Other" val samples
for folder, file in other_val_files:
    src_path = os.path.join(insecta_dir, folder, file)
    dest_path = os.path.join("binary_dataset/val/other", f"{folder}_{file}")
    shutil.copy(src_path, dest_path)

print(f"Copied {len(other_train_files)} 'Other' samples to train set")
print(f"Copied {len(other_val_files)} 'Other' samples to validation set")

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root="binary_dataset/train", transform=train_transform)
val_dataset = ImageFolder(root="binary_dataset/val", transform=val_transform)

print(f"Train dataset size: {len(train_dataset)}, classes: {train_dataset.classes}")
print(f"Validation dataset size: {len(val_dataset)}, classes: {val_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Model
model = mobilenet_v2(weights='DEFAULT')
# Modify the classifier for binary classification
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)  # Binary classification
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# Training function
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    loss = running_loss / len(loader)
    return loss, accuracy

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
    loss = running_loss / len(loader)
    return loss, accuracy, macro_f1

# Training loop
print("\nStarting training...")
best_f1 = -1.0
best_acc = -1.0

for epoch in range(NUM_EPOCHS):
    start = time.time()
    
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
    
    end = time.time()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | F1: {val_f1:.2f}% | "
          f"Time: {end-start:.2f}s")
    
    # Save best model
    if val_f1 > best_f1 or (val_f1 == best_f1 and val_acc > best_acc):
        best_f1 = val_f1
        best_acc = val_acc
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'classes': train_dataset.classes
        }, "models/abaeis_binary_model.pth")
        
        # Export to ONNX
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            "models/abaeis_binary_model.onnx",
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        print(f"Saved best model with F1: {val_f1:.2f}%, Accuracy: {val_acc:.2f}%")

# Quantize the ONNX model
try:
    quantized_model_path = "models/abaeis_binary_model_quantized.onnx"
    quantize_dynamic(
        "models/abaeis_binary_model.onnx",
        quantized_model_path,
        weight_type=QuantType.QUInt8
    )
    
    # Print file sizes
    orig_size = os.path.getsize("models/abaeis_binary_model.onnx") / (1024 * 1024)
    quant_size = os.path.getsize(quantized_model_path) / (1024 * 1024)
    
    print(f"Original model size: {orig_size:.2f} MB")
    print(f"Quantized model size: {quant_size:.2f} MB")
    print(f"Size reduction: {(1 - quant_size/orig_size) * 100:.1f}%")
    
except Exception as e:
    print(f"Quantization failed: {e}")

print("\nTraining complete!")
print(f"Best validation F1-Score: {best_f1:.2f}%")
print(f"Best validation accuracy: {best_acc:.2f}%")
print("Model saved as 'models/abaeis_binary_model.pth'")
print("ONNX model saved as 'models/abaeis_binary_model.onnx'")

# Create a small example script to show how to use the model
# with open("abaeis_inference_example.py", "w") as f:
#     f.write('''
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import onnxruntime as ort

# def load_image(image_path, img_size=160):
#     """Load and preprocess an image for inference"""
#     transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# def pytorch_inference(model_path, image_tensor):
#     """Run inference using PyTorch model"""
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#     classes = checkpoint['classes']
    
#     # Load the model (simplified for this example)
#     from torchvision.models import mobilenet_v2
#     model = mobilenet_v2(num_classes=2)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         _, predicted = torch.max(outputs, 1)
#         probability = torch.nn.functional.softmax(outputs, dim=1)[0]
    
#     return {
#         "class_id": predicted.item(),
#         "class_name": classes[predicted.item()],
#         "probability": probability[predicted.item()].item()
#     }

# def onnx_inference(model_path, image_tensor):
#     """Run inference using ONNX Runtime"""
#     # Convert to numpy for ONNX Runtime
#     image_numpy = image_tensor.numpy()
    
#     # Initialize ONNX Runtime session
#     session = ort.InferenceSession(model_path)
    
#     # Run inference
#     input_name = session.get_inputs()[0].name
#     output = session.run(None, {input_name: image_numpy})
    
#     # Process output
#     probabilities = output[0][0]
#     predicted = probabilities.argmax()
    
#     # Class names from training
#     classes = ["abaeis_nicippe", "other"]
    
#     return {
#         "class_id": predicted,
#         "class_name": classes[predicted],
#         "probability": float(probabilities[predicted])
#     }

# if __name__ == "__main__":
#     # Example usage
#     image_path = "binary_dataset/val/abaeis_nicippe/example.jpg"  # Replace with an actual image path
    
#     # Load image
#     image_tensor = load_image(image_path)
    
#     # Run PyTorch inference
#     print("\\nRunning PyTorch inference:")
#     result = pytorch_inference("models/abaeis_binary_model.pth", image_tensor)
#     print(f"Predicted class: {result['class_name']}")
#     print(f"Confidence: {result['probability']:.4f}")
    
#     # Run ONNX inference
#     print("\\nRunning ONNX inference:")
#     result = onnx_inference("models/abaeis_binary_model.onnx", image_tensor)
#     print(f"Predicted class: {result['class_name']}")
#     print(f"Confidence: {result['probability']:.4f}")
# '''
#     )
# print("Created example inference script: 'abaeis_inference_example.py'")
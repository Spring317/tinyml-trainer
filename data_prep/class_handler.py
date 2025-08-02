import json
import os
from typing import List, Dict, Tuple
import torch

class ClassNameHandler:
    """
    Handles class names and metadata for dataset creation and model evaluation.
    """
    
    def __init__(self, class_counter_path: str = "data_prep/class_counts_sorted.json"):
        self.class_counter_path = class_counter_path
        self.class_names = []
        self.num_classes = 0
        self.class_to_idx = {}
        
    def load_class_info_from_json(self, start_rank: int = 0, number_of_dominant_classes: int = 3) -> Dict:
        """
        Load class information from the class counts JSON file.
        
        Args:
            start_rank: Starting rank for selecting dominant species
            number_of_dominant_classes: Number of main classes (excluding "Other")
            
        Returns:
            Dictionary containing class information
        """
        if not os.path.exists(self.class_counter_path):
            raise FileNotFoundError(f"Class counts file not found: {self.class_counter_path}")
            
        with open(self.class_counter_path, 'r', encoding='utf-8') as f:
            class_counts = json.load(f)
            
        all_species = class_counts['class_counts_sorted']
        
        # Selected species for main classes
        end_rank = start_rank + number_of_dominant_classes
        selected_species = all_species[start_rank:end_rank]
        
        # Build class names list
        class_names = []
        class_to_idx = {}
        
        # Add main classes (0, 1, 2, ...)
        for i, species in enumerate(selected_species):
            class_name = species['class_name']
            class_names.append(class_name)
            class_to_idx[class_name] = i
            
        # Add "Other" class as the last class
        class_names.append("Other")
        class_to_idx["Other"] = len(class_names) - 1
        
        # Store information
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_to_idx = class_to_idx
        
        return {
            "class_names": class_names,
            "num_classes": len(class_names),
            "class_to_idx": class_to_idx,
            "selected_species": selected_species,
            "start_rank": start_rank,
            "end_rank": end_rank
        }
        
    def get_class_names_from_model(self, model_path: str) -> List[str]:
        """
        Extract class information from a trained model.
        
        Args:
            model_path: Path to the saved PyTorch model
            
        Returns:
            List of class names
        """
        try:
            # Load model to check output size
            model = torch.load(model_path, weights_only=False, map_location="cpu")
            
            # Try to get number of classes from model architecture
            num_model_classes = None
            if hasattr(model, "classifier"):
                if hasattr(model.classifier, "linear"):
                    num_model_classes = model.classifier.linear.out_features
                elif hasattr(model.classifier, "out_features"):
                    num_model_classes = model.classifier.out_features
            elif hasattr(model, "fc"):
                num_model_classes = model.fc.out_features
                
            if num_model_classes:
                self.num_classes = num_model_classes
                
                # Generate default class names based on number of classes
                if num_model_classes == 4:
                    # Assume 3 main classes + 1 "Other" class
                    self.class_names = ["Class 0", "Class 1", "Class 2", "Other"]
                elif num_model_classes == 2:
                    # Binary classification
                    self.class_names = ["Dominant Species", "Other"]
                else:
                    # General case
                    self.class_names = [f"Class {i}" for i in range(num_model_classes - 1)] + ["Other"]
                    
                print(f"Detected {num_model_classes} classes from model architecture")
                return self.class_names
            else:
                raise ValueError("Could not determine number of classes from model")
                
        except Exception as e:
            print(f"Warning: Could not extract class info from model: {e}")
            # Fallback to default 4-class setup
            self.class_names = ["Class 0", "Class 1", "Class 2", "Other"]
            self.num_classes = 4
            return self.class_names

    def get_class_info(self, start_rank: int = 0, number_of_dominant_classes: int = 3,
                      model_path: str = None) -> Tuple[List[str], int, Dict[str, int]]:
        """
        Get comprehensive class information, trying JSON first, then model fallback.
        
        Args:
            start_rank: Starting rank for dataset creation
            number_of_dominant_classes: Number of main classes
            model_path: Path to model file (fallback)
            
        Returns:
            Tuple of (class_names, num_classes, class_to_idx)
        """
        try:
            # First try to load from JSON
            class_info = self.load_class_info_from_json(start_rank, number_of_dominant_classes)
            print(f"✅ Loaded class information from {self.class_counter_path}")
            print(f"📊 Classes: {class_info['class_names']}")
            return class_info['class_names'], class_info['num_classes'], class_info['class_to_idx']
            
        except Exception as e:
            print(f"⚠️  Could not load from JSON: {e}")
            
            if model_path:
                print("🔄 Trying to extract class info from model...")
                class_names = self.get_class_names_from_model(model_path)
                class_to_idx = {name: i for i, name in enumerate(class_names)}
                return class_names, self.num_classes, class_to_idx
            else:
                print("🔄 Using default class configuration...")
                # Final fallback
                self.class_names = ["Class 0", "Class 1", "Class 2", "Other"]
                self.num_classes = 4
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
                return self.class_names, self.num_classes, self.class_to_idx
                
    def print_class_summary(self):
        """Print a summary of the class information."""
        print("\n" + "="*60)
        print("📋 CLASS INFORMATION SUMMARY")
        print("="*60)
        print(f"Total number of classes: {self.num_classes}")
        print("Class mapping:")
        for name, idx in self.class_to_idx.items():
            print(f"  {idx}: {name}")
        print("="*60)
        
    def save_class_info(self, output_path: str = "class_info.json"):
        """Save current class information to a JSON file."""
        class_info = {
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(class_info, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Class information saved to {output_path}")

# Convenience function for easy usage
def get_class_info_for_evaluation(start_rank: int = 0, number_of_dominant_classes: int = 3, 
                                 model_path: str = None) -> Tuple[List[str], int, Dict[str, int]]:
    """
    Convenience function to get class information for evaluation.
    
    Returns:
        Tuple of (class_names, num_classes, class_to_idx)
    """
    handler = ClassNameHandler()
    return handler.get_class_info(start_rank, number_of_dominant_classes, model_path)

# if __name__ == "__main__":
#     # Test the class handler
#     handler = ClassNameHandler()
    
#     # Test with default parameters
#     class_names, num_classes, class_to_idx = handler.get_class_info(start_rank=0, number_of_dominant_classes=3)
#     handler.print_class_summary()
#     handler.save_class_info()
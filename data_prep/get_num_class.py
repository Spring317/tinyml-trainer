import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def count_samples_per_class(dataset_path: str) -> Tuple[Dict[str, int], int]:
    """
    Scan through the Insecta dataset and count samples for each class

    Args:
        dataset_path (str): Path to the Insecta dataset directory

    Returns:
        dict: Dictionary with class names and their sample counts
    """

    insecta_path = Path(dataset_path)

    if not insecta_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    class_counts = {}
    total_samples = 0

    print(f"📁 Scanning dataset: {insecta_path}")
    print("=" * 60)

    # Get all subdirectories (classes)
    class_dirs = [d for d in insecta_path.iterdir() if d.is_dir()]
    class_dirs.sort()  # Sort alphabetically

    for class_dir in class_dirs:
        class_name = class_dir.name

        # Count image files in the class directory
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        sample_count = 0
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                sample_count += 1

        class_counts[class_name] = sample_count
        total_samples += sample_count

        print(f"{class_name:<35} {sample_count:>6} samples")

    print("=" * 60)
    print(f"Total classes: {len(class_counts)}")
    print(f"Total samples: {total_samples}")

    return class_counts, total_samples


def save_class_counts_sorted_json(
    class_counts: Dict[int, int], output_file: str
) -> Dict:
    """
    Save class counts to a sorted JSON file with statistics

    Args:
        class_counts (dict): Dictionary with class names and counts
        total_samples (int): Total number of samples
        output_file (str): Output JSON file path

    Returns:
        dict: JSON structure with class counts and statistics
    """

    # Calculate statistics

    # Sort classes by sample count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # Create comprehensive JSON structure
    output_data = {
        "class_counts_sorted": [
            {"class_name": class_name, "sample_count": count, "rank": i + 1}
            for i, (class_name, count) in enumerate(sorted_classes)
        ],
    }

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return output_data

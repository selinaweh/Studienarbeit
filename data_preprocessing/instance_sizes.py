import os
import matplotlib.pyplot as plt
import numpy as np


def categorize_bbox_size(left, top, right, bottom):
    """
    Categorize the size of a bounding box based on its area.
    """
    width = right - left
    height = bottom - top
    area = width * height
    if area > 128**2:
        return "Large"
    elif area > 32**2:
        return "Medium"
    elif area > 16**2:
        return "Small"
    else:
        return "Tiny"


def get_size_distribution(label_dirs, CLASS_NAMES):
    """
    Get the distribution of bounding box sizes for each class.
    """
    size_distribution = {v: {"Tiny": 0, "Small": 0, "Medium": 0, "Large": 0} for v in CLASS_NAMES.values()}

    for label_dir in label_dirs:
        for label_file in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")

                    try:
                        left = float(parts[0])
                        top = float(parts[1])
                        right = float(parts[2])
                        bottom = float(parts[3])
                        class_id = int(parts[4])
                    except (ValueError, IndexError):
                        print(f"Error in row {label_file}: {line.strip()}")
                        continue

                    if class_id in CLASS_NAMES:
                        size_category = categorize_bbox_size(left, top, right, bottom)
                        class_name = CLASS_NAMES[class_id]
                        size_distribution[class_name][size_category] += 1
                    else:
                        print(f"Warning: Unknown class ID {class_id} in file {label_file}")

    return size_distribution


def plot_size_distribution(size_distribution):
    """
    Plot the distribution of bounding box sizes for each class.
    """
    categories = ["Tiny", "Small", "Medium", "Large"]
    class_names = list(size_distribution.keys())

    data = np.array([[size_distribution[cls][cat] for cat in categories] for cls in class_names])
    data = data / data.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    bottom = np.zeros(len(class_names))

    colors = ["#AFCBF0", "#7298DA", "#2E5EBF", "#0F3B85"]

    for i, cat in enumerate(categories):
        ax.bar(class_names, data[:, i], bottom=bottom, label=cat, color=colors[i])
        bottom += data[:, i]

    ax.set_ylim(0, 1)
    ax.set_ylabel("Relative Frequency")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(title="Bounding Box Size", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()



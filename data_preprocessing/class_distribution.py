import os
import yaml
from collections import Counter
import matplotlib.pyplot as plt


def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']


def get_class_distribution(label_dirs):
    """
    Count the occurrences of each class in the label files.
    :param label_dirs: List of directories containing label files.
    :return: Counter object with class IDs as keys and their occurrences as values.
    """
    class_counter = Counter()

    for label_dir in label_dirs:
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.txt'):
                label_path = os.path.join(label_dir, file_name)
                with open(label_path, 'r') as file:
                    for line in file:
                        class_id = int(line.split()[0])
                        class_counter[class_id] += 1

    return class_counter


def plot_class_distribution(class_counter, class_names=None, title="Class Distribution"):
    """
    Create a bar plot of the class distribution.
    :param class_counter: Counter object with class IDs as keys
    :param class_names: Dictionary with class IDs as keys and class names as values
    :param title: Title of the plot
    """

    if class_names:
        classes = [class_names[c] for c in class_counter.keys()]
    else:
        classes = list(class_counter.keys())
    counts = list(class_counter.values())

    plt.figure(figsize=(6, 3))
    plt.bar(classes, counts, tick_label=[f"{c}" for c in classes])
    plt.xlabel("Classes")
    plt.ylabel("Counts")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


#reading .npy files for training
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Define the directory where you saved the processed data
processed_dir = Path("C:/Vamsi/Github/HAR_NN/processed")

# Load 3D NumPy arrays (shape: samples × channels × time_steps)
X_train = np.load(processed_dir / "X_train.npy")
y_train = np.load(processed_dir / "y_train.npy")
X_test = np.load(processed_dir / "X_test.npy")
y_test = np.load(processed_dir / "y_test.npy")

print("Loaded .npy files:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")



# Define label-to-activity mapping
activity_labels = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

def analyze_label_distribution(y, subset_name="train"):
    """
    Prints and plots label distribution with activity names.

    Args:
        y (np.ndarray): 1D array of labels
        subset_name (str): Name of the dataset subset ('train' or 'test')
    """
    label_counts = Counter(y)
    total = len(y)

    print(f"\nLabel Distribution in {subset_name.upper()} Set:")
    for label in sorted(label_counts.keys()):
        activity = activity_labels.get(label, f"Unknown ({label})")
        count = label_counts[label]
        print(f"{activity:20s}: {count} samples ({(count/total)*100:.2f}%)")

    # Prepare labels and values for plotting
    activities = [activity_labels[label] for label in sorted(label_counts.keys())]
    counts = [label_counts[label] for label in sorted(label_counts.keys())]

    # Plot distribution
    plt.figure(figsize=(10, 4))
    plt.bar(activities, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Activity")
    plt.ylabel("Number of Samples")
    plt.title(f"{subset_name.capitalize()} Set Label Distribution")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example usage
analyze_label_distribution(y_train, "train")
analyze_label_distribution(y_test, "test")

# %pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_one_sample_per_class(X: np.ndarray, y: np.ndarray, activity_labels: dict, show_legend=False):
    """
    Plots one random time-series sample for each activity class.

    Args:
        X (np.ndarray): Input signals of shape (samples, 9, 128)
        y (np.ndarray): Labels of shape (samples,)
        activity_labels (dict): Mapping from label index (int) to activity name (str)
        show_legend (bool): Whether to show the channel legend for each subplot
    """
    unique_classes = np.unique(y)
    samples_per_class = []

    # Choose one random sample for each activity
    for label in unique_classes:
        indices = np.where(y == label)[0]
        if indices.size == 0:
            continue
        chosen = np.random.choice(indices)
        samples_per_class.append(chosen)

    # Plotting
    num_classes = len(samples_per_class)
    fig, axs = plt.subplots(num_classes, 1, figsize=(12, 2.5 * num_classes), sharex=True)

    if num_classes == 1:
        axs = [axs]  # Ensure axs is iterable if only one subplot

    for i, idx in enumerate(samples_per_class):
        ax = axs[i]
        for ch in range(X.shape[1]):
            ax.plot(X[idx, ch], alpha=0.8, label=f"Ch {ch+1}")
        label_val = int(y[idx])
        activity_name = activity_labels.get(label_val, f"Unknown ({label_val})")
        ax.set_title(f"Activity: {activity_name} – Label: {label_val} – Sample Index: {idx}")
        ax.set_ylabel("Sensor Value")
        ax.grid(True)
        if show_legend:
            ax.legend(loc="upper right", fontsize="small")

    axs[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.show()


activity_labels = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

plot_one_sample_per_class(X_train, y_train, activity_labels, show_legend=False)




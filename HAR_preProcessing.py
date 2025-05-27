# %pip install numpy
# %pip install pandas
import numpy as np
import pandas as pd
from pathlib import Path



#Includes Saving

import numpy as np
from pathlib import Path

# 9 raw signal names for HAR dataset
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

def load_signals_numpy(base_dir: str, subset: str) -> np.ndarray:
    """
    Loads and stacks the 9 inertial signals for train/test sets using NumPy only.

    Args:
        base_dir (str): Path to the HAR dataset root directory
        subset (str): 'train' or 'test'

    Returns:
        np.ndarray: Shape (samples, 9, 128)
    """
    base_path = Path(base_dir) / subset / "Inertial Signals"
    signals = []

    for signal in SIGNAL_NAMES:
        file_path = base_path / f"{signal}_{subset}.txt"
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing file: {file_path}")
        print(f"Loading: {file_path}")
        data = np.loadtxt(file_path)  # Each file shape: (samples, 128)
        signals.append(data)

    # Stack and reshape to (samples, 9, 128)
    stacked = np.stack(signals, axis=0)  # shape: (9, samples, 128)
    return np.transpose(stacked, (1, 0, 2))  # shape: (samples, 9, 128)

def load_labels_numpy(label_path) -> np.ndarray:
    """
    Loads activity labels as 1D NumPy array (zero-indexed).

    Args:
        label_path (str): Full path to label file

    Returns:
        np.ndarray: Shape (samples,)
    """
    print(f"Loading: {label_path}")
    return np.loadtxt(label_path, dtype=int) - 1  # convert to zero-indexed

# Example usage
folder = "C:/Vamsi/Github/HAR_NN/UCI_HAR_Dataset"
X_train = load_signals_numpy(folder, "train")
y_train = load_labels_numpy("C:/Vamsi/Github/HAR_NN/y_train.txt")
X_test = load_signals_numpy(folder, "test")
y_test = load_labels_numpy("C:/Vamsi/Github/HAR_NN/y_test.txt")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ========== Save to disk ========== #

# Output directory
output_dir = Path("C:/Vamsi/Github/HAR_NN/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save 3D arrays as .npy (preserves structure)
np.save(output_dir / "X_train.npy", X_train)
np.save(output_dir / "y_train.npy", y_train)
np.save(output_dir / "X_test.npy", X_test)
np.save(output_dir / "y_test.npy", y_test)

# Save flattened 2D arrays as .csv (useful for inspection or sklearn-style input)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Column names for CSV
columns = [f"ch{ch}_t{t}" for ch in range(9) for t in range(128)]

df_train = pd.DataFrame(X_train_flat, columns=columns)
df_train["label"] = y_train
df_train.to_csv(output_dir / "X_train_with_labels.csv", index=False)

df_test = pd.DataFrame(X_test_flat, columns=columns)
df_test["label"] = y_test
df_test.to_csv(output_dir / "X_test_with_labels.csv", index=False)

print("Saved all data (npy + csv) to:", output_dir)

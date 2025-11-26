import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

# Paths
data_root = "data/processed/test"
model_path = "checkpoints/best_model.keras"
image_root = "data/AFLW2000"

# Load trained model
print("[INFO] Loading trained ST-GCN model...")
model = tf.keras.models.load_model(model_path, compile=False)

# Collect all test samples
test_files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".npz")]
print(f"[INFO] Found {len(test_files)} test samples")

# Randomly pick one for visualization
sample_path = random.choice(test_files)
sample = np.load(sample_path)
X = sample["X"]
y_true = sample["y"]

# Predict
y_pred = model.predict(np.expand_dims(X, axis=0))
y_pred_class = np.argmax(y_pred, axis=1)[0]

print(f"True label: {y_true}, Predicted label: {y_pred_class}")

# Try to find matching image (based on file name)
base_name = os.path.splitext(os.path.basename(sample_path))[0]
possible_jpg = os.path.join(image_root, base_name.replace(".npz", ".jpg"))
if not os.path.exists(possible_jpg):
    # Fallback to any random image if not found
    jpg_files = [f for f in os.listdir(image_root) if f.endswith(".jpg")]
    possible_jpg = os.path.join(image_root, random.choice(jpg_files))

# Load original image
img = np.array(Image.open(possible_jpg).convert("RGB"))

# Use final frame of X as landmark estimate
landmarks = X[-1]
augmented = landmarks + np.random.normal(0, 0.002, landmarks.shape)  # simulated augmentation

# Normalize coordinates to image space
h, w, _ = img.shape
landmarks_2d = np.clip(landmarks[:, :2] * [w, h], 0, [w - 1, h - 1])
augmented_2d = np.clip(augmented[:, :2] * [w, h], 0, [w - 1, h - 1])

# ---- Plot side-by-side comparison ----
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Left: Original input
axes[0].imshow(img)
axes[0].set_title("Input Image")
axes[0].axis("off")

# Right: Output landmarks (overlayed)
axes[1].imshow(img)
axes[1].scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='r', s=12, label='Estimated (Red)')
axes[1].scatter(augmented_2d[:, 0], augmented_2d[:, 1], c='g', s=12, label='Augmented (Green)')
axes[1].set_title("Output: Landmarks Overlay")
axes[1].axis("off")
axes[1].legend(loc='lower right')

plt.suptitle(f"3D Face Recognition Result  |  True: {y_true}  Pred: {y_pred_class}", fontsize=12)
plt.tight_layout()
plt.show()

# ---- Evaluate on entire test set ----
print("[INFO] Evaluating model on all test data...")
X_test, y_test = [], []
for f in test_files:
    d = np.load(f)
    X_test.append(d["X"])
    y_test.append(d["y"])

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred_all = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_all, digits=4))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Validation Set)")
plt.show()

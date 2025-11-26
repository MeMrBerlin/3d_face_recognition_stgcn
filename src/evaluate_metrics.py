import os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from dataset_aflw2000 import make_dataset

def evaluate(data_root="data/processed", ckpt_dir="checkpoints/best_tf", batch_size=16, num_classes=6):
    test_ds, (J, C) = make_dataset(data_root, split="test", batch_size=batch_size, shuffle=False)
    model = tf.keras.models.load_model(ckpt_dir)
    y_true, y_pred = [], []
    for X, y, _A in test_ds:
        logits = model(X, training=False).numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(logits, axis=1).tolist())
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"Accuracy={acc:.4f}  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed")
    ap.add_argument("--ckpt_dir",  type=str, default="checkpoints/best_tf")
    ap.add_argument("--batch_size",type=int, default=16)
    ap.add_argument("--num_classes", type=int, default=6)
    args = ap.parse_args()
    evaluate(args.data_root, args.ckpt_dir, args.batch_size, args.num_classes)

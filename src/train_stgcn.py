import os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from dataset_aflw2000 import make_dataset
from graph_utils import normalize_adjacency
from stgcn_layers import build_stgcn_model

def train(data_root="data/processed", out_dir="checkpoints",
          epochs=20, batch_size=16, lr=1e-3, num_classes=6):
    os.makedirs(out_dir, exist_ok=True)

    train_ds, (J, C) = make_dataset(data_root, split="train", batch_size=batch_size, shuffle=True)
    test_ds,  _      = make_dataset(data_root, split="test",  batch_size=batch_size, shuffle=False)

    # Peek first batch to get adjacency
    X0, y0, A0 = next(iter(train_ds))
    # A0: (B, J, J) but identical per sample; use first
    A_norm = A0[0].numpy()

    # Build model (input shape: (None, J, C) with time dim T=1)
    model = build_stgcn_model(num_joints=J, in_channels=C, num_classes=num_classes, A_norm=A_norm)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def step_train(X, y):
        # reshape to (B, T, J, C) — our X is (B, T, J, C) already, but be explicit
        with tf.GradientTape() as tape:
            logits = model(X, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits

    @tf.function
    def step_eval(X, y):
        logits = model(X, training=False)
        loss = loss_fn(y, logits)
        return loss, logits

    best_f1 = -1.0
    for epoch in range(1, epochs+1):
        # ---- train
        tr_losses, tr_y, tr_pred = [], [], []
        for X, y, _A in train_ds:
            loss, logits = step_train(X, y)
            tr_losses.append(loss.numpy())
            tr_y.extend(y.numpy().tolist())
            tr_pred.extend(np.argmax(logits.numpy(), axis=1).tolist())
        tr_acc = accuracy_score(tr_y, tr_pred)
        tr_p, tr_r, tr_f1, _ = precision_recall_fscore_support(tr_y, tr_pred, average="macro", zero_division=0)

        # ---- eval
        te_losses, te_y, te_pred = [], [], []
        for X, y, _A in test_ds:
            loss, logits = step_eval(X, y)
            te_losses.append(loss.numpy())
            te_y.extend(y.numpy().tolist())
            te_pred.extend(np.argmax(logits.numpy(), axis=1).tolist())
        te_acc = accuracy_score(te_y, te_pred)
        te_p, te_r, te_f1, _ = precision_recall_fscore_support(te_y, te_pred, average="macro", zero_division=0)

        print(f"[{epoch:03d}] train_loss={np.mean(tr_losses):.4f} acc={tr_acc:.3f} p={tr_p:.3f} r={tr_r:.3f} f1={tr_f1:.3f} | "
              f"val_loss={np.mean(te_losses):.4f} acc={te_acc:.3f} p={te_p:.3f} r={te_r:.3f} f1={te_f1:.3f}")

        if te_f1 > best_f1:
            best_f1 = te_f1
            # model.save(os.path.join(out_dir, "best_tf"))
            model.save(os.path.join(out_dir, "best_model.keras"))
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({"J":J, "C":C, "num_classes":num_classes}, f)

    print("\nFinal classification report (validation):")
    print(classification_report(te_y, te_pred, digits=4))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed")
    ap.add_argument("--out_dir",   type=str, default="checkpoints")
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--batch_size",type=int, default=16)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--num_classes", type=int, default=6)
    args = ap.parse_args()
    train(args.data_root, args.out_dir, args.epochs, args.batch_size, args.lr, args.num_classes)

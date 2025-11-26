import os, json
import numpy as np
import tensorflow as tf

def _load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)        # (T=1, J, C)
    y = int(d["y"])
    A = d["A"].astype(np.float32)        # (J,J)
    return X, y, A

def make_dataset(processed_root, split="train", batch_size=16, shuffle=True):
    with open(os.path.join(processed_root, f"{split}/index.json"), "r") as f:
        files = json.load(f)

    def gen():
        for p in files:
            X, y, A = _load_npz(p)
            yield X, y, A

    # Output signatures: X:(T,J,C), y:(), A:(J,J)
    first_X, _, first_A = _load_npz(files[0])
    T, J, C = first_X.shape
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(T, J, C), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(J, J), dtype=tf.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(batch_size)

    # We want a fixed A passed separately to model build; take from the first batch at runtime.
    return ds, (J, C)

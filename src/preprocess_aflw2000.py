import os, json, glob
import numpy as np
from scipy.io import loadmat
from graph_utils import build_spatial_adjacency, normalize_adjacency
from features_3d import augment_landmarks_midpoints, build_node_features_xyz, stack_time

def _read_mat_landmarks(mat_path):
    """
    Returns (J=68, 3) xyz landmarks and optional pose angles if present.
    Many AFLW2000 .mat files contain 'pt3d_68' (3x68) and 'Pose_Para' (1x7) or 'Pose_Para' (3 angles).
    """
    m = loadmat(mat_path)
    if 'pt3d_68' in m:
        pts = m['pt3d_68']  # often shaped (3,68)
        if pts.shape[0] == 3:
            xyz = pts.T.astype(np.float32)   # (68,3)
        elif pts.shape[1] == 3:
            xyz = pts.astype(np.float32)     # (68,3)
        else:
            raise RuntimeError(f"Unexpected pt3d_68 shape in {mat_path}: {pts.shape}")
    else:
        raise RuntimeError(f"No 'pt3d_68' found in {mat_path}")
    # pose (yaw) if available
    yaw_deg = None
    for key in ['Pose_Para', 'pose_para']:
        if key in m:
            pp = m[key].squeeze()
            # different dumps exist; try typical indexing: [pitch, yaw, roll, ...] in radians or degrees
            # We'll heuristically pick element 1 as yaw.
            yaw = float(pp[1])
            # if in radians convert to degrees (assume |yaw| ~< 1.6 means radians)
            if abs(yaw) < 6.0:
                yaw_deg = yaw * 180.0 / np.pi
            else:
                yaw_deg = yaw
            break
    return xyz, yaw_deg

def _yaw_to_class(yaw_deg, bins=(-60,-30,-10,10,30,60)):
    """
    Map yaw (deg) into 6 classes: far-left, left, slight-left, slight-right, right, far-right
    If yaw is None, return 3 (center) as fallback.
    """
    if yaw_deg is None:
        return 3
    edges = list(bins)
    # bins define boundaries between 7 regions -> we want 6 categories around those values
    # We'll use numpy digitize over [-inf, -60, -30, -10, 10, 30, 60, +inf]
    extended = [-1e9] + list(bins) + [1e9]
    idx = np.digitize([yaw_deg], extended)[0] - 1  # 0..6
    # collapse to 6 classes: map 0->0, 1->1, 2->2, 3->3, 4->4, 5/6->5
    if idx >= 6: idx = 6
    if idx == 6: idx = 5
    return int(idx)

def main(data_root="data/AFLW2000", out_dir="data/processed", split_ratio=0.8, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(data_root, "*.jpg")))
    mats   = sorted(glob.glob(os.path.join(data_root, "*.mat")))
    # index by basename stem
    mat_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mats}
    samples = []
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        if stem in mat_map:
            samples.append((img, mat_map[stem]))
    if not samples:
        raise RuntimeError("No (image, mat) pairs found. Check your data/AFLW2000 layout.")

    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    n_train = int(len(samples) * split_ratio)
    train, test = samples[:n_train], samples[n_train:]

    # Build and save processed tensors per sample
    def process_split(items, subdir):
        out_sub = os.path.join(out_dir, subdir)
        os.makedirs(out_sub, exist_ok=True)
        index = []
        for i, (img_path, mat_path) in enumerate(items):
            xyz, yaw = _read_mat_landmarks(mat_path)     # (68,3), float yaw
            # 2D for adjacency and augmentation
            xy = xyz[:, :2]
            aug_xy = augment_landmarks_midpoints(xy, max_pairs=150)
            all_xy = np.vstack([xy, aug_xy]) if len(aug_xy) else xy
            # For features we only keep original 68 xyz nodes (consistent graph size for training)
            feats = build_node_features_xyz(xyz)         # (68,6)
            # graph adjacency on original 68 nodes using their 2D (xy)
            from graph_utils import build_spatial_adjacency, normalize_adjacency
            A = build_spatial_adjacency(xy)              # (68,68)
            A_norm = normalize_adjacency(A)

            X = stack_time([feats])                      # (T=1, J=68, C=6)
            y = _yaw_to_class(yaw)                       # 0..5
            # save npz
            out_npz = os.path.join(out_sub, f"sample_{i:05d}.npz")
            np.savez_compressed(out_npz, X=X, y=y, A=A_norm, img=img_path, aug_xy=aug_xy, base_xy=xy)
            index.append(out_npz)
        with open(os.path.join(out_sub, "index.json"), "w") as f:
            json.dump(index, f, indent=2)

    process_split(train, "train")
    process_split(test,  "test")
    # top-level manifest
    with open(os.path.join(out_dir, "dataset_index.json"), "w") as f:
        json.dump({"train":"train/index.json", "test":"test/index.json"}, f, indent=2)
    print(f"Done. Processed: train={len(train)}, test={len(test)} to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/AFLW2000")
    ap.add_argument("--out_dir",   type=str, default="data/processed")
    ap.add_argument("--split",     type=float, default=0.8)
    args = ap.parse_args()
    main(args.data_root, args.out_dir, args.split)

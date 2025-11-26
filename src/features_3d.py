import numpy as np
from scipy.spatial import KDTree

def augment_landmarks_midpoints(landmarks_xy, max_pairs=150, seed=42):
    """
    Simple augmentation: pick random landmark pairs; add their midpoints in 2D.
    landmarks_xy: (J,2)
    returns aug_xy: (M,2)
    """
    rng = np.random.default_rng(seed)
    J = landmarks_xy.shape[0]
    pairs = set()
    for _ in range(min(max_pairs, J*2)):
        i = int(rng.integers(0, J))
        j = int(rng.integers(0, J))
        if i != j:
            if i > j: i, j = j, i
            pairs.add((i, j))
    mids = []
    for (i, j) in pairs:
        p = (landmarks_xy[i] + landmarks_xy[j]) * 0.5
        mids.append(p)
    if len(mids) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(mids, dtype=np.float32)

def build_node_features_xyz(landmarks_xyz):
    """
    Make per-node features from 3D landmarks:
      - raw (x,y,z) normalized per face (zero-mean, unit-std)
      - deltas to centroid (x',y',z')
    Output: (J, C=6)
    """
    x = landmarks_xyz.astype(np.float32)  # (J,3)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x - mu) / sd
    feat = np.concatenate([x_norm, x - mu], axis=1)  # (J,6)
    return feat.astype(np.float32)

def stack_time(features_list):
    """
    Given per-frame features [(J,C), ...], return (T,J,C).
    For AFLW2000 we use T=1.
    """
    X = np.stack(features_list, axis=0)  # (T, J, C)
    return X

import numpy as np
from scipy.spatial import Delaunay

def build_spatial_adjacency(landmarks_xy):
    """
    Delaunay triangulation over 2D landmarks to define edges.
    landmarks_xy: (J,2) float32
    returns A: (J,J) binary adjacency
    """
    tri = Delaunay(landmarks_xy)
    J = landmarks_xy.shape[0]
    A = np.zeros((J, J), dtype=np.float32)
    for simplex in tri.simplices:
        i, j, k = simplex
        A[i, j] = A[j, i] = 1.0
        A[j, k] = A[k, j] = 1.0
        A[i, k] = A[k, i] = 1.0
    np.fill_diagonal(A, 0.0)
    return A

def normalize_adjacency(A):
    """
    Return A_norm = D^{-1/2} (A + I) D^{-1/2}
    """
    J = A.shape[0]
    A_hat = A + np.eye(J, dtype=np.float32)
    deg = A_hat.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)

def build_adjacency_matrix(landmarks_xy=None, num_nodes=68):
    """
    Compatibility wrapper so older scripts using build_adjacency_matrix() keep working.
    If landmarks are provided, use Delaunay triangulation to build adjacency.
    Otherwise, create a simple normalized chain graph.
    """
    if landmarks_xy is not None:
        A = build_spatial_adjacency(landmarks_xy)
    else:
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        np.fill_diagonal(A, 1.0)

    A_norm = normalize_adjacency(A)
    return A_norm

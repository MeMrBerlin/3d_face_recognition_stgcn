# import os, glob
# import cv2
# import numpy as np
# from scipy.io import loadmat
# from features_3d import augment_landmarks_midpoints

# def read_landmarks_xy_from_mat(mat_path):
#     m = loadmat(mat_path)
#     if 'pt3d_68' not in m:
#         raise RuntimeError("No 'pt3d_68' in MAT.")
#     pts = m['pt3d_68']
#     if pts.shape[0] == 3:
#         xyz = pts.T.astype(np.float32)        # (68,3)
#     else:
#         xyz = pts.astype(np.float32)
#     return xyz[:, :2]

# def draw_fig2(image_path, mat_path, out_path, radius=2):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise RuntimeError(f"Cannot read {image_path}")
#     base_xy = read_landmarks_xy_from_mat(mat_path)     # (68,2)
#     aug_xy  = augment_landmarks_midpoints(base_xy)     # (M,2)

#     vis = img.copy()
#     # red: base
#     for (x,y) in base_xy.astype(int):
#         cv2.circle(vis, (x,y), radius, (0,0,255), -1)
#     # green: augmented
#     for (x,y) in aug_xy.astype(int):
#         cv2.circle(vis, (x,y), radius, (0,255,0), -1)
#     cv2.imwrite(out_path, vis)
#     print(f"Saved: {out_path}")

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--image", required=True)
#     ap.add_argument("--mat", required=True)
#     ap.add_argument("--out", default="results/fig2_demo/demo.png")
#     args = ap.parse_args()
#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     draw_fig2(args.image, args.mat, args.out)

# import os
# import cv2
# import numpy as np
# from scipy.io import loadmat
# from features_3d import augment_landmarks_midpoints

# # ----------------------------
# # Helper functions
# # ----------------------------

# def read_landmarks_xy_from_mat(mat_path):
#     """Read 2D landmarks from AFLW2000 .mat file."""
#     m = loadmat(mat_path)
#     if 'pt3d_68' not in m:
#         raise RuntimeError("No 'pt3d_68' in MAT.")
#     pts = m['pt3d_68']
#     if pts.shape[0] == 3:
#         xyz = pts.T.astype(np.float32)  # (68,3)
#     else:
#         xyz = pts.astype(np.float32)
#     return xyz[:, :2]


# def draw_fig2(image_path, mat_path, out_path, radius=2):
#     """Draw red (base) and green (augmented) landmarks."""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise RuntimeError(f"Cannot read {image_path}")
#     base_xy = read_landmarks_xy_from_mat(mat_path)  # (68,2)
#     aug_xy = augment_landmarks_midpoints(base_xy)   # (M,2)

#     vis = img.copy()
#     # Red: base landmarks
#     for (x, y) in base_xy.astype(int):
#         cv2.circle(vis, (x, y), radius, (0, 0, 255), -1)
#     # Green: augmented landmarks
#     for (x, y) in aug_xy.astype(int):
#         cv2.circle(vis, (x, y), radius, (0, 255, 0), -1)

#     # Save visualization
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     cv2.imwrite(out_path, vis)
#     print(f"✅ Saved visualization: {out_path}")


# # ----------------------------
# # Main interactive section
# # ----------------------------

# if __name__ == "__main__":
#     print("=== 3D Face Landmark Visualization (ST-GCN Figure 2 Demo) ===")
#     img_num = input("Enter image number (e.g., 00013): ").strip()

#     # Paths
#     img_path = f"data/AFLW2000/image{img_num}.jpg"
#     mat_path = f"data/AFLW2000/image{img_num}.mat"
#     out_dir = "results/fig2_demo"
#     os.makedirs(out_dir, exist_ok=True)

#     # Auto-increment output filename
#     existing = sorted([f for f in os.listdir(out_dir) if f.startswith("demo") and f.endswith(".png")])
#     next_idx = len(existing) + 1
#     out_path = os.path.join(out_dir, f"demo{next_idx}.png")

#     # Run visualization
#     if not os.path.exists(img_path):
#         print(f"❌ Image not found: {img_path}")
#     elif not os.path.exists(mat_path):
#         print(f"❌ MAT file not found: {mat_path}")
#     else:
#         print(f"[INFO] Processing image{img_num}.jpg ...")
#         draw_fig2(img_path, mat_path, out_path)


# import os
# import cv2
# import numpy as np
# from scipy.io import loadmat

# # Optional: Mediapipe support
# try:
#     import mediapipe as mp
#     HAS_MEDIAPIPE = True
# except ImportError:
#     HAS_MEDIAPIPE = False
#     print("[WARN] Mediapipe not found. Install it using `pip install mediapipe` to support random images.")

# from features_3d import augment_landmarks_midpoints


# # ------------------------------------------------
# # Helper Functions
# # ------------------------------------------------
# def read_landmarks_xy_from_mat(mat_path):
#     """Reads (68, 2) landmarks from .mat file"""
#     m = loadmat(mat_path)
#     if 'pt3d_68' not in m:
#         raise RuntimeError("No 'pt3d_68' in MAT file.")
#     pts = m['pt3d_68']
#     xyz = pts.T if pts.shape[0] == 3 else pts
#     return xyz[:, :2].astype(np.float32)


# def extract_landmarks_from_image(image_path):
#     """Extracts (468, 2) face landmarks using Mediapipe FaceMesh if .mat is missing."""
#     if not HAS_MEDIAPIPE:
#         raise ImportError("Mediapipe not installed. Run `pip install mediapipe` to use this feature.")

#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

#     img = cv2.imread(image_path)
#     if img is None:
#         raise RuntimeError(f"Cannot read {image_path}")

#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)
#     if not results.multi_face_landmarks:
#         raise RuntimeError("No face detected in the image.")

#     h, w, _ = img.shape
#     landmarks = np.array([[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark], dtype=np.float32)
#     print(f"[INFO] Extracted {len(landmarks)} landmarks from {os.path.basename(image_path)}")
#     return landmarks


# def draw_fig2(image_path, mat_path, out_path, radius=2):
#     """Draws both base (red) and augmented (green) landmarks."""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise RuntimeError(f"Cannot read {image_path}")

#     # Try loading landmarks
#     if os.path.exists(mat_path):
#         print(f"[INFO] Using landmarks from {mat_path}")
#         base_xy = read_landmarks_xy_from_mat(mat_path)
#     else:
#         print(f"[INFO] .mat file not found for {image_path}. Using automatic landmark detection.")
#         base_xy = extract_landmarks_from_image(image_path)

#     # Augment landmarks
#     aug_xy = augment_landmarks_midpoints(base_xy)

#     vis = img.copy()
#     # Red: base landmarks
#     for (x, y) in base_xy.astype(int):
#         cv2.circle(vis, (x, y), radius, (0, 0, 255), -1)
#     # Green: augmented landmarks
#     for (x, y) in aug_xy.astype(int):
#         cv2.circle(vis, (x, y), radius, (0, 255, 0), -1)

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     cv2.imwrite(out_path, vis)
#     print(f"✅ Saved visualization to {out_path}")


# # ------------------------------------------------
# # Interactive CLI
# # ------------------------------------------------
# if __name__ == "__main__":
#     data_dir = "data"
#     output_dir = "results/fig2_demo"
#     os.makedirs(output_dir, exist_ok=True)

#     # Ask user for image number or filename
#     img_input = input("Enter image number (e.g., 00013) or full filename: ").strip()

#     # 🔹 Updated logic for flexible path detection
#     if os.path.exists(os.path.join(data_dir, f"{img_input}.jpg")):
#         image_path = os.path.join(data_dir, f"{img_input}.jpg")
#         mat_path = os.path.splitext(image_path)[0] + ".mat"
#     elif os.path.exists(os.path.join(data_dir, f"image{img_input}.jpg")):
#         image_path = os.path.join(data_dir, f"image{img_input}.jpg")
#         mat_path = os.path.join(data_dir, f"image{img_input}.mat")
#     elif os.path.exists(os.path.join(data_dir, "AFLW2000", f"image{img_input}.jpg")):
#         image_path = os.path.join(data_dir, "AFLW2000", f"image{img_input}.jpg")
#         mat_path = os.path.join(data_dir, "AFLW2000", f"image{img_input}.mat")
#     else:
#         raise FileNotFoundError(f"Could not find image for '{img_input}' in data/ or data/AFLW2000/")

#     # Automatically increment output filename
#     existing = sorted([f for f in os.listdir(output_dir) if f.startswith("demo") and f.endswith(".png")])
#     next_idx = len(existing) + 1
#     out_path = os.path.join(output_dir, f"demo{next_idx}.png")

#     print(f"[INFO] Processing {image_path}")
#     draw_fig2(image_path, mat_path, out_path)


import os
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.spatial import Delaunay
from features_3d import augment_landmarks_midpoints

# ----------------------------
# Helper: Read 2D landmarks (68 pts)
# ----------------------------
def read_landmarks_xy_from_mat(mat_path):
    m = loadmat(mat_path)
    if 'pt3d_68' not in m:
        raise RuntimeError("No 'pt3d_68' in MAT.")
    pts = m['pt3d_68']

    if pts.shape[0] == 3:
        xyz = pts.T.astype(np.float32)  # (68,3)
    else:
        xyz = pts.astype(np.float32)

    return xyz[:, :2]   # return only (x,y)


# ---------------------------------------------------
# FIXED: Moderate augmentation (not all pair midpoints)
# Generates 68 base + 68 midpoints = 136 total
# ---------------------------------------------------
def augment_landmarks_midpoints(base_xy):
    """
    Create moderate-density augmentation:
    Only midpoints between each consecutive landmark pair.
    This avoids the 2000+ points explosion.
    """
    mids = []
    n = len(base_xy)

    for i in range(n):
        j = (i + 1) % n        # next landmark (wrap-around)
        mids.append((base_xy[i] + base_xy[j]) / 2.0)

    mids = np.array(mids, dtype=np.float32)

    return np.vstack([base_xy, mids])   # (136,2)


# ----------------------------
# Helper: Build Delaunay mesh
# ----------------------------
def build_edges(landmarks_xy):
    tri = Delaunay(landmarks_xy)
    edges = set()

    for simplex in tri.simplices:
        i, j, k = simplex
        edges.update([(i, j), (j, k), (i, k)])

    return list(edges)


# ----------------------------
# Visualization
# ----------------------------
def draw_fig2(image_path, mat_path, out_path, radius=2):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")

    # Base + Augmented landmarks
    base_xy = read_landmarks_xy_from_mat(mat_path)      # (68,2)
    aug_xy = augment_landmarks_midpoints(base_xy)       # (136,2)

    vis = img.copy()

    # Build mesh edges for both sets
    base_edges = build_edges(base_xy)
    aug_edges = build_edges(aug_xy)

    # Draw base mesh (red)
    for (i, j) in base_edges:
        pt1 = tuple(base_xy[i].astype(int))
        pt2 = tuple(base_xy[j].astype(int))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1)

    # Draw augmented mesh (green)
    for (i, j) in aug_edges:
        pt1 = tuple(aug_xy[i].astype(int))
        pt2 = tuple(aug_xy[j].astype(int))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    # Draw landmarks
    for (x, y) in base_xy.astype(int):
        cv2.circle(vis, (x, y), radius, (0, 0, 255), -1)

    for (x, y) in aug_xy.astype(int):
        cv2.circle(vis, (x, y), radius, (0, 255, 0), -1)

    # Save output
    cv2.imwrite(out_path, vis)
    print(f"✅ Saved visualization with mesh: {out_path}")

    # REMOVED: No freezing
    # cv2.imshow("Facial Mesh Visualization", vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    os.makedirs("results/fig2_demo", exist_ok=True)

    # Ask user for image number
    img_num = input("Enter image number (e.g., 00013): ").strip()

    image_path = f"data/AFLW2000/image{img_num}.jpg"
    mat_path = f"data/AFLW2000/image{img_num}.mat"

    # Auto increment demo filename
    existing = sorted(os.listdir("results/fig2_demo"))
    demo_id = len(existing) + 1

    out_path = f"results/fig2_demo/demo{demo_id}.png"

    draw_fig2(image_path, mat_path, out_path)





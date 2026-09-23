import glob, os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree 

DRIVE_DIR = "/home/sarath/ekf-pose-fusion/the-kitti-pose-estimation-dataset/data/2011_09_26/2011_09_26_drive_0002_sync/" 

def load_scan(drive_dir, index):
    path = os.path.join(drive_dir, "velodyne_points", "data", f"{index:010d}.bin")
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]          # keep x, y, z ; drop reflectance

def preprocess(pts, z_min=-1.5, z_max=2.0, max_range=40.0, step=4):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    mask  = (z > z_min) & (z < z_max)      # above the road, below the sky
    mask &= (np.hypot(x, y) < max_range)   # drop far, sparse points
    pts = pts[mask]
    pts = pts[::step]                      # keep every 4th point (thin out)
    return pts[:, :2]                      # top-down: return x, y only

def best_fit_transform_2d(P, Q):
    """Best rotate+slide that maps matched points P onto matched points Q.
       (Standard 'point set alignment' via SVD — treat it as a tool.)"""
    muP, muQ = P.mean(axis=0), Q.mean(axis=0)      # centers of each set
    Pc, Qc = P - muP, Q - muQ                      # center both
    W = Qc.T @ Pc                                  # 2x2 cross-covariance
    U, S, Vt = np.linalg.svd(W)
    R = U @ Vt                                     # best rotation
    if np.linalg.det(R) < 0:                       # guard against a mirror flip
        U[:, -1] *= -1
        R = U @ Vt
    t = muQ - R @ muP                              # best slide
    return R, t

def icp_2d(src, dst, max_iter=40, tol=1e-5):
    """Move src onto dst. Returns total rotation angle, slide, final gap."""
    tree = cKDTree(dst)                            # for fast nearest-point lookups
    R_total, t_total = np.eye(2), np.zeros(2)
    moved = src.copy()
    prev = None
    for _ in range(max_iter):
        dists, idx = tree.query(moved)             # 1) match: nearest dst point
        R, t = best_fit_transform_2d(moved, dst[idx])  # 2) fit
        moved = (R @ moved.T).T + t                # 3) apply
        R_total = R @ R_total                       #    fold into the running total
        t_total = R @ t_total + t
        err = dists.mean()                          # 4) average gap
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err
    phi = np.arctan2(R_total[1, 0], R_total[0, 0])  # total turn angle
    return t_total, phi, err

# All the LIDAR points
# A = load_scan(DRIVE_DIR, 0)   # scan at step 0
# B = load_scan(DRIVE_DIR, 1)   # scan at step 1 (0.1 s later)

# Ground points are masked. 
# The LIDAR is placed at 1.7m high
A = preprocess(load_scan(DRIVE_DIR, 0))
B = preprocess(load_scan(DRIVE_DIR, 1))
print("points after cleaning:", len(A), len(B))

t, phi, err = icp_2d(B, A)                 # move current onto previous
print("slide (m):", t, "  turn (deg):", np.degrees(phi), "  final gap (m):", err)

plt.figure(figsize=(8, 8))
plt.scatter(A[:, 0], A[:, 1], s=1, label="scan 0 (cleaned)")
plt.scatter(B[:, 0], B[:, 1], s=1, label="scan 1 (cleaned)")
plt.axis("equal"); plt.legend()
plt.xlabel("x forward [m]"); plt.ylabel("y left [m]")
plt.title("cleaned scans (ground removed, thinned)")
plt.savefig("two_scans_clean.png", dpi=150); 
plt.show()

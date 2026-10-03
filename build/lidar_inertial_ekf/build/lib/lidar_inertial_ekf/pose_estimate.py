"""
KITTI LiDAR + IMU pose estimation (EKF sensor fusion).

Pipeline (all built from scratch, plain numpy + scipy):
  1. Load data (oxts = IMU/GPS/ground-truth, velodyne = LiDAR scans).
  2. IMU-only dead reckoning        -> drifts (baseline).
  3. EKF: IMU predict + position-fix correct  -> beats dead reckoning.
  4. LiDAR odometry (ICP scan matching)        -> motion from raw scans.
  (Next step, not here yet: feed LiDAR motion into the EKF instead of the GPS fix.)

Run:  python3 pose_fusion.py
"""

import glob
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DRIVE_DIR = "/home/sarath/ekf-pose-fusion/the-kitti-pose-estimation-dataset/data/2011_09_26/2011_09_26_drive_0002_sync/"

# oxts column map (0-indexed)
IDX_LAT, IDX_LON = 0, 1
IDX_YAW = 5              # heading, 0 = EAST, CCW positive (rad)
IDX_VN, IDX_VE = 6, 7    # velocity north / east (world frame, m/s)
IDX_VF = 8              # forward speed (m/s)
IDX_AF = 14             # forward acceleration, car frame (m/s^2)
IDX_WU = 22             # yaw rate about up axis (rad/s)

# EKF tuning
P0_DIAG = np.array([0.5, 0.5, 0.1, 0.5])   # initial 1-sigma [x, y, theta, v]
GYRO_STD = 0.01
ACCEL_STD = 0.05

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_oxts(drive_dir):
    """Load every oxts frame + its timestamp, sorted by frame index."""
    files = sorted(glob.glob(os.path.join(drive_dir, "oxts", "data", "*.txt")))
    frames = np.array([np.loadtxt(f) for f in files])

    times = []
    with open(os.path.join(drive_dir, "oxts", "timestamps.txt")) as fh:
        for line in fh:
            # KITTI stamps have 9 fractional digits (ns); %f only takes 6.
            times.append(datetime.strptime(line.strip()[:26], "%Y-%m-%d %H:%M:%S.%f"))
    t = np.array([(ti - times[0]).total_seconds() for ti in times])
    return frames, t


def latlon_to_xy(lats, lons):
    """Local flat-earth projection: lat/lon (deg) -> east/north meters, origin at frame 0."""
    R = 6378137.0
    x_east = R * np.radians(lons - lons[0]) * np.cos(np.radians(lats[0]))
    y_north = R * np.radians(lats - lats[0])
    return x_east, y_north


def load_scan(drive_dir, index):
    """Load one LiDAR scan as (N, 3) points x,y,z (drop reflectance)."""
    path = os.path.join(drive_dir, "velodyne_points", "data", f"{index:010d}.bin")
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def preprocess(pts, z_min=-1.5, z_max=2.0, max_range=40.0, step=4):
    """Clean a scan for matching: drop ground + far points, thin, keep top-down x,y."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    mask = (z > z_min) & (z < z_max)
    mask &= (np.hypot(x, y) < max_range)
    pts = pts[mask]
    pts = pts[::step]
    return pts[:, :2]


# ===========================================================================
# 2. IMU-ONLY DEAD RECKONING (baseline)  -- world-velocity model
# ===========================================================================
def dead_reckon(frames, t):
    n = len(frames)
    xs, ys = np.zeros(n), np.zeros(n)

    theta = frames[0, IDX_YAW]
    x, y = 0.0, 0.0
    vx = frames[0, IDX_VE]   # east velocity
    vy = frames[0, IDX_VN]   # north velocity

    xs[0], ys[0] = x, y
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        af = frames[k - 1, IDX_AF]
        wu = frames[k - 1, IDX_WU]

        theta = theta + wu * dt
        ax_w = af * np.cos(theta)
        ay_w = af * np.sin(theta)
        vx = vx + ax_w * dt
        vy = vy + ay_w * dt
        x = x + vx * dt
        y = y + vy * dt

        xs[k], ys[k] = x, y
    return xs, ys


# ===========================================================================
# 3. ICP SCAN MATCHING (2D)
# ===========================================================================
def best_fit_transform_2d(P, Q):
    """Best rotate+slide mapping matched points P onto matched points Q (via SVD)."""
    muP, muQ = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - muP, Q - muQ
    W = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(W)
    R = U @ Vt
    if np.linalg.det(R) < 0:            # guard against a mirror flip
        U[:, -1] *= -1
        R = U @ Vt
    t = muQ - R @ muP
    return R, t


def icp_2d(src, dst, max_iter=40, tol=1e-5):
    """Move src onto dst. Returns (slide t, turn phi, final gap)."""
    tree = cKDTree(dst)
    R_total, t_total = np.eye(2), np.zeros(2)
    moved = src.copy()
    prev = None
    for _ in range(max_iter):
        dists, idx = tree.query(moved)                  # 1) match
        R, t = best_fit_transform_2d(moved, dst[idx])   # 2) fit
        moved = (R @ moved.T).T + t                     # 3) apply
        R_total = R @ R_total                            #    fold into total
        t_total = R @ t_total + t
        err = dists.mean()                               # 4) average gap
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err
    phi = np.arctan2(R_total[1, 0], R_total[0, 0])
    return t_total, phi, err


# ===========================================================================
# 4. EXTENDED KALMAN FILTER  (state = [x, y, theta, v])
# ===========================================================================
def predict_filter(frames, t):
    """Predict-only filter (teaching version): moves state + grows cloud P.
       Trajectory equals dead reckoning; uncertainty just climbs."""
    n = len(frames)
    xs, ys, unc = np.zeros(n), np.zeros(n), np.zeros(n)

    x, y = 0.0, 0.0
    theta = frames[0, IDX_YAW]
    v = frames[0, IDX_VF]
    P = np.diag([0.5, 0.5, 0.1, 0.5]) ** 2

    xs[0], ys[0] = x, y
    unc[0] = np.trace(P)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        af = frames[k - 1, IDX_AF]
        wu = frames[k - 1, IDX_WU]

        theta = theta + wu * dt
        v = v + af * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt

        F = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt, np.cos(theta) * dt],
            [0.0, 1.0, v * np.cos(theta) * dt, np.sin(theta) * dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        Q = np.diag([1e-6, 1e-6, (0.01 * dt) ** 2, (0.05 * dt) ** 2])
        P = F @ P @ F.T + Q

        xs[k], ys[k] = x, y
        unc[k] = np.trace(P)
    return xs, ys, unc


def ekf(frames, t, gt_x, gt_y, fix_every=10, fix_noise=1.5, seed=0):
    """IMU predict + position-fix correct. The fix is a noisy sample of the
       true path (a training-wheels stand-in for LiDAR)."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    xs, ys, unc = np.zeros(n), np.zeros(n), np.zeros(n)

    state = np.array([0.0, 0.0, frames[0, IDX_YAW], frames[0, IDX_VF]])
    P = np.diag([0.5, 0.5, 0.1, 0.5]) ** 2
    H = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0]])
    R = np.diag([fix_noise ** 2, fix_noise ** 2])

    xs[0], ys[0] = state[0], state[1]
    unc[0] = np.trace(P)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        af = frames[k - 1, IDX_AF]
        wu = frames[k - 1, IDX_WU]

        # ---- predict ----
        x, y, theta, v = state
        theta = theta + wu * dt
        v = v + af * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        state = np.array([x, y, theta, v])

        F = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt, np.cos(theta) * dt],
            [0.0, 1.0, v * np.cos(theta) * dt, np.sin(theta) * dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        Q = np.diag([1e-6, 1e-6, (0.01 * dt) ** 2, (0.05 * dt) ** 2])
        P = F @ P @ F.T + Q

        # ---- correct (only on fix steps) ----
        if k % fix_every == 0:
            z = np.array([gt_x[k], gt_y[k]]) + rng.normal(0, fix_noise, size=2)
            surprise = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ surprise
            P = (np.eye(4) - K @ H) @ P

        xs[k], ys[k] = state[0], state[1]
        unc[k] = np.trace(P)
    return xs, ys, unc


# ===========================================================================
# 5. LiDAR ODOMETRY (chain ICP motions into a path)
# ===========================================================================
def lidar_odometry(drive_dir, n_scans, theta0):
    xs, ys = np.zeros(n_scans), np.zeros(n_scans)
    x, y, theta = 0.0, 0.0, theta0

    prev = preprocess(load_scan(drive_dir, 0))
    for k in range(1, n_scans):
        cur = preprocess(load_scan(drive_dir, k))
        t_step, dphi, gap = icp_2d(cur, prev)
        dx_b, dy_b = t_step
        # rotate the car-frame step into the world and add it (dead reckoning)
        x += np.cos(theta) * dx_b - np.sin(theta) * dy_b
        y += np.sin(theta) * dx_b + np.cos(theta) * dy_b
        theta += dphi
        xs[k], ys[k] = x, y
        prev = cur
        if k % 10 == 0:
            print(f"  lidar frame {k}/{n_scans}  gap={gap:.3f}")
    return xs, ys

def lidar_diagnostics(drive_dir, n_scans, frames, gt_x, gt_y):
    prev = preprocess(load_scan(drive_dir, 0))
    l_fwd, g_fwd, l_dphi, g_dphi = [], [], [], []
    for k in range(1, n_scans):
        cur = preprocess(load_scan(drive_dir, k))
        t_step, dphi, gap = icp_2d(cur, prev)
        l_fwd.append(np.hypot(*t_step))                         # LiDAR forward step
        g_fwd.append(np.hypot(gt_x[k]-gt_x[k-1], gt_y[k]-gt_y[k-1]))  # true forward step
        l_dphi.append(np.degrees(dphi))                          # LiDAR turn
        d = frames[k, IDX_YAW] - frames[k-1, IDX_YAW]
        g_dphi.append(np.degrees((d + np.pi) % (2*np.pi) - np.pi))    # true turn
        prev = cur

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.plot(g_fwd, label="truth"); a.plot(l_fwd, label="LiDAR")
    a.set_title("forward distance per step [m]"); a.legend(); a.set_xlabel("frame")
    b.plot(g_dphi, label="truth"); b.plot(l_dphi, label="LiDAR")
    b.set_title("turn per step [deg]"); b.legend(); b.set_xlabel("frame")
    plt.tight_layout(); plt.savefig("lidar_diag.png", dpi=150); plt.show()

def rmse(ax, ay, bx, by):
    return float(np.sqrt(np.mean((ax - bx) ** 2 + (ay - by) ** 2)))

def lidar_per_step_turn(drive_dir, n_scans):
    """LiDAR turn (rad) between each pair of scans, via ICP."""
    dphi = np.zeros(n_scans)
    prev = preprocess(load_scan(drive_dir, 0))
    for k in range(1, n_scans):
        cur = preprocess(load_scan(drive_dir, k))
        _, phi, _ = icp_2d(cur, prev)
        dphi[k] = phi
        prev = cur
    return dphi

def ekf_lidar(frames, t, lidar_dphi, heading_noise_deg=1.0):
    n = len(frames)
    xs, ys, unc = np.zeros(n), np.zeros(n), np.zeros(n)
    state = np.array([0.0, 0.0, frames[0, IDX_YAW], frames[0, IDX_VF]])
    P = np.diag([0.5, 0.5, 0.1, 0.5]) ** 2

    H = np.array([[0.0, 0.0, 1.0, 0.0]])                 # measures theta only
    R = np.array([[np.radians(heading_noise_deg) ** 2]])
    theta_lidar = frames[0, IDX_YAW]                      # accumulated LiDAR heading

    xs[0], ys[0] = state[0], state[1]; unc[0] = np.trace(P)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        af = frames[k - 1, IDX_AF]; wu = frames[k - 1, IDX_WU]

        # ---- predict (unchanged) ----
        x, y, theta, v = state
        theta = theta + wu * dt
        v = v + af * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        state = np.array([x, y, theta, v])
        F = np.array([
            [1.0, 0.0, -v*np.sin(theta)*dt, np.cos(theta)*dt],
            [0.0, 1.0,  v*np.cos(theta)*dt, np.sin(theta)*dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        Q = np.diag([1e-6, 1e-6, (0.01*dt)**2, (0.05*dt)**2])
        P = F @ P @ F.T + Q

        # ---- correct with LiDAR heading ----
        if k < len(lidar_dphi):
            theta_lidar += lidar_dphi[k]                 # accumulate LiDAR heading
            surprise = np.array([theta_lidar - state[2]])
            surprise[0] = (surprise[0] + np.pi) % (2*np.pi) - np.pi   # WRAP the angle
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ surprise
            state[2] = (state[2] + np.pi) % (2*np.pi) - np.pi         # keep theta wrapped
            P = (np.eye(4) - K @ H) @ P

        xs[k], ys[k] = state[0], state[1]; unc[k] = np.trace(P)
    return xs, ys, unc

def ekf_predict(state, P, af, wu, dt):
    """Shared EKF predict step: move the state, grow the cloud P."""
    x, y, theta, v = state
    theta = theta + wu * dt
    v = v + af * dt
    x = x + v * np.cos(theta) * dt
    y = y + v * np.sin(theta) * dt
    state = np.array([x, y, theta, v])
    F = np.array([
        [1.0, 0.0, -v*np.sin(theta)*dt, np.cos(theta)*dt],
        [0.0, 1.0,  v*np.cos(theta)*dt, np.sin(theta)*dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Q = np.diag([1e-6, 1e-6, (GYRO_STD*dt)**2, (ACCEL_STD*dt)**2])
    return state, F @ P @ F.T + Q

# ===========================================================================
# 6. MAIN
# ===========================================================================
def main():
    frames, t = load_oxts(DRIVE_DIR)
    gt_x, gt_y = latlon_to_xy(frames[:, IDX_LAT], frames[:, IDX_LON])

    print("sanity: vf vs |vn,ve| =",
          frames[0, IDX_VF], np.hypot(frames[0, IDX_VN], frames[0, IDX_VE]))

    # --- IMU dead reckoning ---
    dr_x, dr_y = dead_reckon(frames, t)

    # --- EKF (IMU + position fix) ---
    ex, ey, eunc = ekf(frames, t, gt_x, gt_y)

    # --- LiDAR odometry ---
    scan_files = sorted(glob.glob(os.path.join(DRIVE_DIR, "velodyne_points", "data", "*.bin")))
    n_scans = len(scan_files)
    lx, ly = lidar_odometry(DRIVE_DIR, n_scans, frames[0, IDX_YAW])
    m = min(n_scans, len(gt_x))

    lidar_dphi = lidar_per_step_turn(DRIVE_DIR, n_scans)
    fx, fy, _ = ekf_lidar(frames, t, lidar_dphi)
    print(f"  LiDAR-inertial EKF : {rmse(fx[:m], fy[:m], gt_x[:m], gt_y[:m]):.3f}")


    # --- scores ---
    print("\n=== RMSE vs ground truth (meters) ===")
    print(f"  IMU dead reckoning : {rmse(dr_x, dr_y, gt_x, gt_y):.3f}")
    print(f"  EKF (IMU + fix)    : {rmse(ex, ey, gt_x, gt_y):.3f}")
    print(f"  LiDAR odometry     : {rmse(lx[:m], ly[:m], gt_x[:m], gt_y[:m]):.3f}")

    # --- plots ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    a1.plot(gt_x, gt_y, "k-", label="ground truth")
    a1.plot(dr_x, dr_y, color="salmon", label="IMU dead reckoning")
    a1.plot(ex, ey, "g-", label="EKF (IMU + fix)", linewidth=2)
    a1.plot(lx[:m], ly[:m], "b--", label="LiDAR odometry")
    a1.plot(fx[0], fy[1], "m-", label="LiDAR-inertial EKF", linewidth=2)

    a1.axis("equal"); a1.legend(); a1.set_xlabel("east [m]"); a1.set_ylabel("north [m]")
    a1.set_title("trajectories vs ground truth")
    # add to the trajectory plot:
    a1.plot(fx, fy, "m-", label="LiDAR-inertial EKF", linewidth=2)

    a2.plot(t, eunc)
    a2.set_xlabel("t [s]"); a2.set_ylabel("uncertainty (trace of P)")
    a2.set_title("EKF uncertainty (saw-tooth)")

    plt.tight_layout()
    plt.savefig("pose_fusion.png", dpi=150)
    plt.show()

    lidar_diagnostics(DRIVE_DIR, n_scans, frames, gt_x, gt_y)

if __name__ == "__main__":
    main()
import glob, os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# --- your locked sensor map (0-indexed array positions) ---
IDX_LAT, IDX_LON = 0, 1
IDX_YAW = 5              # heading, 0 = EAST, CCW positive (rad)
IDX_VN, IDX_VE = 6, 7    # velocity north, east (world frame, m/s)
IDX_AF = 14             # forward acceleration, body x (m/s^2)
IDX_WU = 22             # yaw rate about up axis (rad/s)

DRIVE_DIR = "/home/sarath/ekf-pose-fusion/the-kitti-pose-estimation-dataset/data/2011_09_26/2011_09_26_drive_0002_sync/" 

def load_oxts(drive_dir):
    """Load every oxts frame + its timestamp, sorted by frame index."""
    files = sorted(glob.glob(os.path.join(drive_dir, "oxts", "data", "*.txt")))
    frames = np.array([np.loadtxt(f) for f in files])

    times = []
    with open(os.path.join(drive_dir, "oxts", "timestamps.txt")) as fh:
        for line in fh:
            # KITTI stamps have 9 fractional digits (ns); %f only takes 6,
            # so trim to microseconds. This is the classic parsing trap.
            times.append(datetime.strptime(line.strip()[:26], "%Y-%m-%d %H:%M:%S.%f"))
    t = np.array([(ti - times[0]).total_seconds() for ti in times])
    return frames, t


def latlon_to_xy(lats, lons):
    """Local flat-earth projection: lat/lon (deg) -> east/north meters, origin at frame 0."""
    R = 6378137.0
    x_east  = R * np.radians(lons - lons[0]) * np.cos(np.radians(lats[0]))
    y_north = R * np.radians(lats - lats[0])
    return x_east, y_north


def dead_reckon(frames, t):
    """IMU-only dead reckoning. World frame: x=east, y=north."""
    n = len(frames)
    xs, ys = np.zeros(n), np.zeros(n)

    # --- initialization from frame 0 ---
    theta = frames[0, IDX_YAW]        # heading from EAST (rad)
    x, y = 0.0, 0.0                   # start defines the world origin
    vx = frames[0,7]   # TODO: seed EAST velocity from frame 0 (which channel is already 'east'?)
    vy = frames[0,6]   # TODO: seed NORTH velocity from frame 0

    xs[0], ys[0] = x, y
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        af = frames[k - 1, IDX_AF]    # forward accel this step
        wu = frames[k - 1, IDX_WU]    # yaw rate this step

        # TODO 1 -- update heading from yaw rate:      theta = ...
        theta = theta + wu * dt
        # TODO 2 -- rotate body accel (af, 0) into world using theta:
        #           (remember: corrected matrix, top row uses  -sin)
        #           ax_w = ... ; ay_w = ...
        ax_w = af * np.cos(theta)
        ay_w = af * np.sin(theta)
        # TODO 3 -- integrate world accel into velocity: vx += ... ; vy += ...
        vx = vx + ax_w * dt
        vy = vy + ay_w * dt
        # TODO 4 -- integrate velocity into position:    x  += ... ; y  += ...
        x = x + vx * dt
        y = y + vy * dt

        xs[k], ys[k] = x, y
    return xs, ys


if __name__ == "__main__":
    frames, t = load_oxts(DRIVE_DIR)

    # quick sanity check before trusting anything (yesterday's lesson):
    print("frame0 vf vs |vn,ve|:",
          frames[0, 8], np.hypot(frames[0, IDX_VN], frames[0, IDX_VE]))

    gt_x, gt_y = latlon_to_xy(frames[:, IDX_LAT], frames[:, IDX_LON])
    dr_x, dr_y = dead_reckon(frames, t)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(gt_x, gt_y, "k-", label="ground truth")
    a1.plot(dr_x, dr_y, "r-", label="IMU dead reckoning")
    a1.axis("equal"); a1.legend(); a1.set_xlabel("east [m]"); a1.set_ylabel("north [m]")
    a1.set_title("trajectory")

    err = np.hypot(dr_x - gt_x, dr_y - gt_y)
    a2.plot(t, err); a2.set_xlabel("t [s]"); a2.set_ylabel("position error [m]")
    a2.set_title("drift over time")

    plt.tight_layout(); plt.savefig("dead_reckon.png", dpi=150); plt.show()

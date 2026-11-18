#!/usr/bin/env python3
"""
Simple stereo robot localization using techniques from the course exercises:

- Week 1: HSV colour thresholding, morphology and contours
- Week 2: Lucas-Kanade sparse optical flow
- Week 3: Stereo matching / disparity
- Week 7: Standard matrix Kalman filter

The existing stereo calibration and the working SGBM parameters are retained.
"""

import glob
import re
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ============================================================
# 1. Parameters
# ============================================================

LEFT_DIR = "3_curved_bend/left"
RIGHT_DIR = "3_curved_bend/right"
CALIB_FILE = "stereo_calib.npz"
OUTPUT_DIR = Path("robot_estimation_results")

# 0.001 if the stereo calibration translation is in millimetres;
# 1.0 if it is already in metres.
CALIB_LENGTH_TO_METRES = 0.001

FPS = 10.0

# Yellow colour range (HSV)
LOWER_YELLOW = np.array([14, 75, 50], dtype=np.uint8)
UPPER_YELLOW = np.array([36, 255, 255], dtype=np.uint8)

# Simple contour tracking
MIN_CONTOUR_AREA = 3
MAX_CONTOUR_AREA = 3000
MAX_TRACK_DISTANCE = 180
INITIAL_SEARCH = (300, 200, 700, 600)  # x0, y0, x1, y1
MOTION_THRESHOLD = 16
MOTION_DILATION_SIZE = 11

# Keep the working stereo matcher settings
NUM_DISPARITIES = 96
BLOCK_SIZE = 5
MIN_VALID_DISPARITY = 1.0
MIN_VALID_PIXELS = 12
DEPTH_PATCH_SCALE = 2.5
MAX_DISPARITY_MAD = 12.0

# Keep the working Kalman noise parameters
PROCESS_NOISE = 0.05
MEASUREMENT_NOISE = 2.5

# Lucas-Kanade parameters (Week 2 exercise)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30, 0.01))

# Plotly axes each span 20 metres:
# X: -10 to +10 m (left is negative), depth: 0 to 20 m,
# vertical camera coordinate: 0 to 20 m.
AXIS_HALF_RANGE = 10.0
AXIS_MAX = 10.0
MAX_CLOUD_POINTS = 25000

# Camera pose correction: negative X-axis rotation for downward pitch, 
# and positive translation in Y to shift the height coordinate.
PITCH_ROT_X_DEG = 5.0    # Adjust angle (e.g., -10 to -30 degrees)
TRANSLATION_Y = -1.0     # Positive translation in metres


# ============================================================
# 2. Load image names and stereo calibration
# ============================================================

def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def image_files(folder):
    files = []
    for extension in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(str(Path(folder) / extension)))
    return sorted(files, key=natural_key)


left_files = image_files(LEFT_DIR)
right_files = image_files(RIGHT_DIR)

if not left_files or len(left_files) != len(right_files):
    raise RuntimeError(
        f"Expected matching stereo images: {len(left_files)} left, "
        f"{len(right_files)} right."
    )

if len(left_files) < 2:
    raise RuntimeError("At least two stereo pairs are required.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with np.load(CALIB_FILE) as calib:
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    R1, R2 = calib["R1"], calib["R2"]
    P1, P2, Q = calib["P1"], calib["P2"], calib["Q"]
    W, H = map(int, calib["image_size"])

mapL1, mapL2 = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, (W, H), cv2.CV_32FC1
)
mapR1, mapR2 = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, (W, H), cv2.CV_32FC1
)


def load_pair(index):
    left = cv2.imread(left_files[index])
    right = cv2.imread(right_files[index])

    if left is None or right is None:
        raise RuntimeError(f"Could not read stereo pair {index}.")

    if left.shape[1::-1] != (W, H) or right.shape[1::-1] != (W, H):
        raise RuntimeError(
            f"Image size must match calibration size {(W, H)}."
        )

    left = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
    right = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)
    return left, right


# ============================================================
# 3. Robot detection: HSV threshold + contours (Week 1)
# ============================================================

def detect_robot(image, previous_position, previous_gray=None):
    """Detect the yellow robot, using frame motion to reject static yellow objects."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # Smaller opening preserves the robot's yellow region; closing joins gaps.
    yellow = cv2.morphologyEx(
        yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    yellow = cv2.morphologyEx(
        yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # On later frames, prefer yellow pixels that also changed between frames.
    if previous_gray is not None:
        difference = cv2.absdiff(gray, previous_gray)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        _, motion = cv2.threshold(
            difference, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        motion = cv2.dilate(
            motion,
            np.ones((MOTION_DILATION_SIZE, MOTION_DILATION_SIZE), np.uint8),
            iterations=1
        )
        moving_yellow = cv2.bitwise_and(yellow, motion)
        # If the motion mask is too restrictive, retain colour candidates;
        # temporal proximity below still prevents jumping to unrelated objects.
        if cv2.countNonZero(moving_yellow) >= MIN_CONTOUR_AREA:
            mask = moving_yellow
        else:
            mask = yellow
    else:
        mask = yellow

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    x0, y0, x1, y1 = INITIAL_SEARCH

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            continue
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        if not (H * 0.5 <= cy <= H * 0.95):
            continue

        if previous_position is None:
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            # Prefer compact, reasonably large yellow candidates near the ROI centre.
            roi_centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            distance = np.linalg.norm(np.array([cx, cy]) - roi_centre)
            score = distance / (1.0 + min(area, 500.0) / 100.0)
        else:
            distance = np.linalg.norm(np.array([cx, cy]) - previous_position)
            if distance > MAX_TRACK_DISTANCE:
                continue
            # Prefer nearby components, with a mild preference for larger regions.
            score = distance / (1.0 + min(area, 500.0) / 150.0)

        candidates.append((score, contour, np.array([cx, cy])))

    if not candidates:
        return None, None, mask, gray

    candidates.sort(key=lambda item: item[0])
    _, contour, center = candidates[0]
    return center, cv2.boundingRect(contour), mask, gray


# ============================================================
# 4. Stereo disparity and 3D position
# ============================================================

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=8 * BLOCK_SIZE**2,
    P2=32 * BLOCK_SIZE**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def disparity_map(left, right):
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    return stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0


def estimate_position(left, right, center, bbox):
    disparity = disparity_map(left, right)

    cx, cy = center
    x, y, box_w, box_h = bbox

    # Sample a small area around the detected yellow robot part.
    half_w = max(12, int(box_w * DEPTH_PATCH_SCALE))
    half_h = max(10, int(box_h * DEPTH_PATCH_SCALE))

    xa = max(0, int(cx - half_w))
    xb = min(W, int(cx + half_w))
    ya = max(0, int(cy - half_h))
    yb = min(H, int(cy + half_h))

    patch = disparity[ya:yb, xa:xb]
    values = patch[
        np.isfinite(patch) & (patch > MIN_VALID_DISPARITY)
    ]

    if len(values) < MIN_VALID_PIXELS:
        return None, disparity, None

    d = float(np.median(values))
    mad = float(np.median(np.abs(values - d)))

    if mad > MAX_DISPARITY_MAD:
        return None, disparity, d

    # Reproject pixel position and disparity through calibrated Q.
    X, Y, Z, scale = Q @ np.array([cx, cy, d, 1.0])

    if abs(scale) < 1e-10:
        return None, disparity, d

    xyz = np.array([X, Y, Z]) / scale
    xyz *= CALIB_LENGTH_TO_METRES

    if not np.isfinite(xyz).all() or xyz[2] <= 0:
        return None, disparity, d

    return xyz, disparity, d


# ============================================================
# 5. Standard matrix Kalman filter (Week 7)
# ============================================================

def kalman_filter(measurements, dt):
    """
    Constant-velocity Kalman filter.
    State: [x, vx, y, vy, z, vz]
    Measurement: [x, y, z]
    """
    count = len(measurements)
    filtered = np.full((count, 3), np.nan)

    valid = np.isfinite(measurements).all(axis=1)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return filtered

    F = np.array([
        [1, dt, 0,  0,  0,  0],
        [0,  1, 0,  0,  0,  0],
        [0,  0, 1, dt,  0,  0],
        [0,  0, 0,  1,  0,  0],
        [0,  0, 0,  0,  1, dt],
        [0,  0, 0,  0,  0,  1]
    ], dtype=float)

    Hm = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ], dtype=float)

    Qm = np.eye(6) * PROCESS_NOISE**2
    Rm = np.eye(3) * MEASUREMENT_NOISE**2
    I = np.eye(6)

    first = valid_indices[0]
    state = np.array([
        measurements[first, 0], 0,
        measurements[first, 1], 0,
        measurements[first, 2], 0
    ], dtype=float).reshape(6, 1)

    covariance = np.eye(6)

    for i in range(first, count):
        if i > first:
            state = F @ state
            covariance = F @ covariance @ F.T + Qm

        if valid[i]:
            z = measurements[i].reshape(3, 1)
            error = z - Hm @ state
            S = Hm @ covariance @ Hm.T + Rm
            K = covariance @ Hm.T @ np.linalg.inv(S)

            state = state + K @ error
            covariance = (I - K @ Hm) @ covariance

        filtered[i] = state[[0, 2, 4], 0]

    return filtered


# ============================================================
# 6. Process all stereo pairs
# ============================================================

n_frames = len(left_files)
uv_measurements = np.full((n_frames, 2), np.nan)
xyz_measurements = np.full((n_frames, 3), np.nan)
disparities = np.full(n_frames, np.nan)
boxes = [None] * n_frames

previous_center = None
previous_gray = None

first_cloud_left = None
first_cloud_disparity = None

for i in range(n_frames):
    left, right = load_pair(i)

    center, bbox, yellow_mask, current_gray = detect_robot(
        left, previous_center, previous_gray
    )

    if center is not None:
        uv_measurements[i] = center
        boxes[i] = bbox
        previous_center = center

        xyz, disp, d = estimate_position(
            left, right, center, bbox
        )

        if d is not None:
            disparities[i] = d

        if xyz is not None:
            xyz_measurements[i] = xyz

            if first_cloud_left is None:
                first_cloud_left = left.copy()
                first_cloud_disparity = disp.copy()

    previous_gray = current_gray

    if (i + 1) % 25 == 0 or i == n_frames - 1:
        print(f"Processed {i + 1}/{n_frames} frames")

valid_uv = np.isfinite(uv_measurements).all(axis=1)
valid_xyz = np.isfinite(xyz_measurements).all(axis=1)

if valid_uv.sum() < 3:
    raise RuntimeError(
        "Too few detections. Check HSV thresholds and INITIAL_SEARCH."
    )

if valid_xyz.sum() < 3:
    raise RuntimeError(
        "Too few valid 3D measurements. Check disparity and calibration."
    )

print(f"2D detections: {valid_uv.sum()}/{n_frames}")
print(f"3D measurements: {valid_xyz.sum()}/{n_frames}")

xyz_filtered = kalman_filter(xyz_measurements, 1.0 / FPS)

# --- Apply the same rotation and translation to the trajectory ---
theta = np.radians(PITCH_ROT_X_DEG)
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
xyz_filtered = (xyz_filtered @ R_x.T)
xyz_filtered[:, 1] += TRANSLATION_Y
# -----------------------------------------------------------------

valid_3d_indices = np.where(
    np.isfinite(xyz_filtered).all(axis=1)
)[0]
first_3d = int(valid_3d_indices[0])
last_3d = int(valid_3d_indices[-1])

trajectory = xyz_filtered[first_3d:last_3d + 1]


# ============================================================
# 7. Lucas-Kanade optical flow video (Week 2)
# ============================================================

video_path = OUTPUT_DIR / "optical_flow.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (W, H))

print("Generating optical flow video...")

for i in range(n_frames - 1):
    left1, _ = load_pair(i)
    left2, _ = load_pair(i + 1)

    flow_image = left2.copy()

    # If the robot is detected in both consecutive frames, compute optical flow
    if valid_uv[i] and valid_uv[i+1]:
        gray1 = cv2.cvtColor(left1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(left2, cv2.COLOR_BGR2GRAY)

        # Detect corners around the robot in frame i
        cx, cy = uv_measurements[i]
        x0, x1 = max(0, int(cx - 90)), min(W, int(cx + 90))
        y0, y1 = max(0, int(cy - 65)), min(H, int(cy + 65))

        roi = gray1[y0:y1, x0:x1]
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=100, qualityLevel=0.005, minDistance=1
        )

        if corners is not None and len(corners) > 0:
            corners[:, 0, 0] += x0
            corners[:, 0, 1] += y0

            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, corners, None, **LK_PARAMS
            )

            if new_points is not None:
                good_old = corners[status.ravel() == 1]
                good_new = new_points[status.ravel() == 1]

                for old, new in zip(good_old, good_new):
                    # Flatten the arrays to ensure a 2-element sequence (x, y)
                    p1 = tuple(np.round(old.flatten()).astype(int))
                    p2 = tuple(np.round(new.flatten()).astype(int))

                    if np.linalg.norm(np.array(p2) - np.array(p1)) > 1:
                        cv2.line(flow_image, p1, p2, (0, 255, 0), 2)
                        cv2.circle(flow_image, p2, 3, (0, 0, 255), -1)

    video_writer.write(flow_image)

video_writer.release()


# ============================================================
# 10. 2D trajectory plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

# Keep the original camera/calibration coordinate convention.
ax.plot(
    trajectory[:, 0], trajectory[:, 2],
    linewidth=2, label="Kalman-filtered trajectory"
)
ax.scatter(
    xyz_measurements[valid_xyz, 0],
    xyz_measurements[valid_xyz, 2],
    s=12, alpha=0.4, label="Raw measurements"
)
ax.scatter(
    trajectory[0, 0], trajectory[0, 2],
    s=60, label="Start"
)
ax.scatter(
    trajectory[-1, 0], trajectory[-1, 2],
    s=60, label="End"
)

ax.set_title("Top-down robot trajectory")
ax.set_xlabel("X (m)")
ax.set_ylabel("Camera depth (m)")
ax.grid(True)
ax.axis("equal")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "trajectory_2d.png", dpi=200)
plt.close(fig)


# ============================================================
# 11. 3D point cloud and trajectory (20 m per axis)
# ============================================================

if first_cloud_left is not None:
    points = cv2.reprojectImageTo3D(
        first_cloud_disparity, Q
    )
    points *= CALIB_LENGTH_TO_METRES

    # --- Negative X Rotation + Positive Y Translation ---
    theta = np.radians(PITCH_ROT_X_DEG)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    shape = points.shape
    flat_points = points.reshape(-1, 3)
    
    # Safely suppress warnings from NaN values in invalid disparity pixels
    with np.errstate(invalid='ignore'):
        flat_points = flat_points @ R_x.T
        
    points = flat_points.reshape(shape)
    points[:, 1] += TRANSLATION_Y  # Apply positive translation to Y (height)
    # ---------------------------------------------------

    cloud_mask = (
        (first_cloud_disparity > MIN_VALID_DISPARITY)
        & np.isfinite(points).all(axis=2)
        & (points[:, :, 2] > 0)
        & (-points[:, :, 1] >= 0)  # Filters out any remaining points below height 0
    )

    cloud = points[cloud_mask]
    colors = cv2.cvtColor(
        first_cloud_left, cv2.COLOR_BGR2RGB
    )[cloud_mask]

    if len(cloud) > MAX_CLOUD_POINTS:
        indices = np.random.default_rng(42).choice(
            len(cloud), MAX_CLOUD_POINTS, replace=False
        )
        cloud = cloud[indices]
        colors = colors[indices]

    color_values = [
        f"rgb({r},{g},{b})" for r, g, b in colors
    ]

    fig3d = go.Figure()

    if len(cloud) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 2],
            z=-cloud[:, 1],
            mode="markers",
            marker=dict(
                size=1.5,
                color=color_values,
                opacity=0.55
            ),
            name="Stereo point cloud"
        ))

    fig3d.add_trace(go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 2],
        z=-trajectory[:, 1],
        mode="lines+markers",
        line=dict(width=6),
        marker=dict(size=3),
        name="Robot trajectory"
    ))

    fig3d.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=5, symbol="diamond"),
        name="Left camera origin"
    ))

    fig3d.update_layout(
        title="3D Stereo Map and Robot Trajectory",
        scene=dict(
            xaxis=dict(
                title="X (m)",
                range=[-AXIS_HALF_RANGE, AXIS_HALF_RANGE]
            ),
            yaxis=dict(
                title="Camera depth (m)",
                range=[0, 2*AXIS_MAX]
            ),
            zaxis=dict(
                title="Height / vertical coordinate -Y (m)",
                range=[-2, AXIS_MAX]
            ),
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=45)
    )

    fig3d.write_html(OUTPUT_DIR / "trajectory_3d.html")


# ============================================================
# 12. Summary
# ============================================================

print("\nProcessing completed.")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
print(" - trajectory_2d.png")
print(" - trajectory_3d.html")
if flow_image is not None:
    print(" - optical_flow.png")

print(
    "\nThe 3D viewer axes each span 20 metres. "
    "The displayed depth axis is 0 to 20 m; X and height are -10 to +10 m.")
# perception-finalproject

Final project for 34759 Perception for Autonomous Systems (DTU, Fall 2026).
Calibrate a stereo camera system, then detect and track the Husky A200 robot in 2D and 3D across four sequences.

## Task distribution

| Task | Person |
| --- | --- |
| Calibration and intro | Tim |
| Sequence 1 (no obstacle) | Jana |
| Sequence 2 (obstacle) | Hania |
| Sequence 3 (curved bend) | Caio |
| Sequence 4 (circle) | Einar |

## Repo structure

```
perception-finalproject/
├── data/                     # images, not in git (see below)
├── results/                  # outputs: plots, videos, etc.
│   └── stereo_calib.npz      # saved calibration, used by stereo.py
├── calibration.ipynb         # stereo calibration and rectification
├── stereo.py                 # load the calibration and rectify image pairs
└── README.md
```

Notebooks go in the root folder.

## Data

The `data/` folder is in `.gitignore`, so it is never pushed. Everyone has to add the data themselves.

Download the images from DTU Learn and put them in `data/` like this:

```
data/
├── 0_calib/
│   ├── left/
│   └── right/
├── 1_no_obstacle/
│   ├── left/
│   └── right/
├── 2_obstacle/
│   ├── left/
│   └── right/
├── 3_curved_bend/
│   ├── left/
│   └── right/
└── 4_circle/
    ├── left/
    └── right/
```

## Calibration

Done in `calibration.ipynb` with the images in `0_calib`. The results are saved in `results/stereo_calib.npz`, so you don't need to run the calibration yourself.

- Checkerboard: 7 x 5 inner corners, squares of 80 mm
- Stereo reprojection error: about 0.21 px
- Baseline: about 119 mm
- All 3D values are in millimetres

### How to use it

Your notebook has to be in the root folder so the import and the path work.

```python
import cv2
from stereo import load_rectification, rectify

# once, at the top of the notebook
mapL, mapR, Q = load_rectification()

# for every frame
imgL = cv2.imread("data/2_obstacle/left/000000.png")
imgR = cv2.imread("data/2_obstacle/right/000000.png")
rectL, rectR = rectify(imgL, imgR, mapL, mapR)
```

- Always compute disparity on the rectified images (`rectL`, `rectR`), not the raw ones.
- `Q` turns a disparity map into 3D points: `points = cv2.reprojectImageTo3D(disparity, Q)`. The disparity has to be in pixels as a float. OpenCV's `StereoBM` and `StereoSGBM` return disparity multiplied by 16, so divide by 16 first.
- If the calibration is changed, only `results/stereo_calib.npz` changes. Pull and your code keeps working.

## Sequences

Each sequence has the same three steps:

1. Compute disparity maps and reconstruct the 3D structure of the scene from the rectified stereo pairs.
2. Detect the Husky A200 robot in the images.
3. Track the robot over time in 2D and 3D, including estimating its position when it is fully occluded.

### Sequence 1: `1_no_obstacle` (Jana)

Constant speed, constant distance to the camera, no occlusions.

- [ ] Disparity maps and 3D reconstruction
- [ ] Detect the robot
- [ ] Track the robot in 2D and 3D

### Sequence 2: `2_obstacle` (Hania)

Constant speed, constant distance to the camera, with occlusions.

- [ ] Disparity maps and 3D reconstruction
- [ ] Detect the robot
- [ ] Track the robot in 2D and 3D, including while it is occluded

### Sequence 3: `3_curved_bend` (Caio)

Curved path, distance to the camera changes over time.

- [ ] Disparity maps and 3D reconstruction
- [ ] Detect the robot
- [ ] Track the robot in 2D and 3D

### Sequence 4: `4_circle` (Einar)

Circular path, distance to the camera changes over time.

- [ ] Disparity maps and 3D reconstruction
- [ ] Detect the robot
- [ ] Track the robot in 2D and 3D

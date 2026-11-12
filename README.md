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
├── data/              # images, not in git (see below)
├── results/           # outputs: plots, videos, etc.
├── calibration.ipynb  # stereo calibration and rectification
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

Calibrate the stereo camera system with the images in `0_calib` and rectify the input images.

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

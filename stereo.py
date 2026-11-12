"""Load the saved stereo calibration and rectify image pairs."""
import cv2
import numpy as np


def load_rectification(path="results/stereo_calib.npz"):
    """Return the rectification maps for both cameras and Q (disparity -> 3D)."""
    c = np.load(path)
    size = tuple(int(v) for v in c["image_size"])
    mapL = cv2.initUndistortRectifyMap(c["mtxL"], c["distL"], c["R1"], c["P1"], size, cv2.CV_32FC1)
    mapR = cv2.initUndistortRectifyMap(c["mtxR"], c["distR"], c["R2"], c["P2"], size, cv2.CV_32FC1)
    return mapL, mapR, c["Q"]


def rectify(img_left, img_right, mapL, mapR):
    """Undistort and rectify one stereo pair."""
    rect_left = cv2.remap(img_left, mapL[0], mapL[1], cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, mapR[0], mapR[1], cv2.INTER_LINEAR)
    return rect_left, rect_right
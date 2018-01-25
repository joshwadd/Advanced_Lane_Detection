# Advanced Lane Detection

## Project Overview

This project builds and algorithmic pipeline of computer vision techniques to detect road lanes from the front facing camera on an autonomous car, and the identify the properties of lane required for then controlling the car. The algorithmic pipeline is implemented in Python using OpenCV and is made up of the following files.

| File                                | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Code/calibration.py`      | `CameraCalibration` class used to remove any distortions in the image associated with the imaging hardware. |
| `Code/edge_detection.py`     | Set of gradient and colour space transformation routines used to detect pixels associated with lane lines. |
| `Code/perspective.py`   | `PerspectiveTransformation` class to transform the perspective of the front facing camera image to a overhead lane view. |
| `Code/line.py` | `Line` class to represent a single lane line |dete
| `Code/window.py`        | `Window` class to represent a single scanning window used to detect pixels likely associated with lane lines.|
| `Code/lane_detection.py`      | `LaneDectection` class implementing a lane detection processing pipeline to a consecutive series of frames from a video. |

## Lane Detection Pipeline

The processing pipeline used for the lane detection and identification of lane properties is composed of the following key steps. These are further outlined in the subsequent sections below.

1. **Camera calibration** :  Use of imaging hardware in practice often produces distortions in the obtained images that can change the perceived location and size of objects in the image. For accurately detecting and then measuring the properties of the lane as it appears in the real world, this distortion must be corrected for. and removed.
2. **Edge detection** : Pixels associated with the lane lines are identified in the image using gradient and colour space transforms and returned via a binary mask image of these pixels.
3. **Perspective Transform** : The perspective of the binary image is transformed to give an over head 'birdseye' view of the lane lines. This makes it easier to detect and the lane in the following steps.
4. **Line detection** : With an overhead perspective of the binary image of lane pixels we
5.  **Fit Polynomial** :
6. **Measuring lane parameters** :


<!--stackedit_data:
eyJoaXN0b3J5IjpbNDkxNjMyODgxXX0=
-->
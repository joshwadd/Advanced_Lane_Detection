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
4. **Line detection** : With an overhead perspective of the binary image of lane pixels, the pixels associated with the left and right lane lines are detected. This can be done initially by measuring a histogram of the pixel distribution in the overhead view and fitting a series of sliding windows. For future frames this can be done by searching in regions close to previously detected lines.
5.  **Fit Polynomial** : A second order polynomial model is fit to through the pixels associated with the left and right lane lines.
6. **Measuring lane parameters** : The model fit through the lane lines can be used to calculate the radius of curvature of the lane, and the position of the car in the lane.

## Camera Calibration

Code for the camera calibration stage can be found with the `CameraCalibration` class in`Code/calibration.py`. 

Any camera using a lens to obtain a 2D image in a 3D environment can produce both radial and tangential distortions in the obtained image. This can change the apparent shape, size and location of objects in the image, and thus must be removed for accurate and useful visual information to be extracted.

The distortions can be quantified and then removed by comparing multiple distorted images of a simple object with known spatial properties. A typically object used for this purpose is a chess board as it has simple shape of known size that are simple to detect due to the high black and white contrast.

OpenCV has a set of routine that all





<!--stackedit_data:
eyJoaXN0b3J5IjpbLTYwMTAzMzc0N119
-->
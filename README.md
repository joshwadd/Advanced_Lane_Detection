# Advanced Lane Detection

## Project Overview

This project builds and algorithmic pipeline of computer vision techniques to detect road lanes from the front facing camera on an autonomous car, and the identify the properties of lane required for then controlling the car. The algorithmic pipeline is implemented in Python using OpenCV and is made up of the following files.

| File                                | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Code/calibration.py`      | Class implementing camera calibration based on the set of calibration images. |
| `Code/edge_detection.py`     | Set of gradient and colour space transformation routines to detect pixels associated with lane lines. |
| `Code/perspective.py`   | Class to implement a perspective transformation to an overhead lane view. |
| `Code/line.py` | Class to represent a single lane line |
| `Code/window.py`        | |
| `Code/lane_detection.py`      | `Window` class representing a scanning window used to detect points likely to represent lines. |
 Implements lane tracking by applying a processing pipeline to consecutive frames in a video.
`Window` class representing a scanning window used to detect points likely to represent lines.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMjMzNTQxMTFdfQ==
-->
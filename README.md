# Advanced Lane Detection

## Project Overview

This project builds and algorithmic pipeline of computer vision techniques to detect road lanes from the front facing camera on an autonomous car, and the identify the properties of lane required for then controlling the car. The algorithmic pipeline is implemented in Python using OpenCV and is made up of the following files.

| File                                | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Code/camera.py`      | Implements camera calibration based on the set of calibration images. |
| `source/lanetracker/tracker.py`     | Implements lane tracking by applying a processing pipeline to consecutive frames in a video. |
| `source/lanetracker/gradients.py`   | Set of edge-detecting routines based on gradients and color. |
| `source/lanetracker/perspective.py` | Set of perspective transformation routines. |
| `source/lanetracker/line.py`        | `Line` class representing a single lane boundary line. |
| `source/lanetracker/window.py`      | `Window` class representing a scanning window used to detect points likely to represent lines. |
| `source/vehicletracker/features.py` | Implements feature extraction pipeline for vehicle tracking. |
| `source/vehicletracker/tracker.py`  | Implements surrounding vehicles tracking by applying a processing pipeline to consecutive frames in a video. |
| `source/vehicletracker/utility.py`  | Set of convenient logging routines. |

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NzY5NDc3MDJdfQ==
-->
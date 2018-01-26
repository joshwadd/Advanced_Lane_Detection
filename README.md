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

OpenCV has a set of routines for the quantification and then removal of camera distortion. First the coordinates of the chess board corners in 3D space are held in the variable `corner_points`. As the chess board will remain fixed in the same location in 3D space for each image this variable will remain the same for each calibration image.

```python
#Prepare the object points belonging to the chess board corners
#eg (0,0,0), (0,0,1), (0,0 2) etc
corner_points = np.zeros((nx*ny,3), np.float32)
corner_points[:,:,2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
```

For each calibration image the location of the chess board corners must be found. This can be done with the OpenCV function `cv2.findChessboardCorners()` 

![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/corner_detection.png?raw=true)

This function is used on every calibration image making a list of corner coordinates for each image and alongside a list of the corners in 3D space. 

```python
obj_points = []
img_points = []
for image in calibration_image:
	ret, corners = cv2.findChessboardCorners(image, (nx,ny), None)
	if ret:
		obj_points.append(corner_points)
		img_points.append(corners)
	
```

With the lists of object and image points it is now possible to compute the camera calibration matrix and distortion coefficients using the function `cv2.calibrateCamera()`.  The calibration matrix and distortion coefficients are computed and stored within the class on initialisation.


```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
```
Once the calibration matrix and distortion coefficients are computed, distortions can be removed from images using the `cv2.undistort` function.  This is implemented in the `CameraCalibration` as 

```python
def undistort(self, image):
	return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
```



 

![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/original.png?raw=true)![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/undistorted.png?raw=true)

## Edge Detection

To identify the pixels belonging to the lane lines in the image, combinations of gradient thresholding and colour space threshholding techniques are used.

This is done by breaking the edge detecting process into two tracks. The first filters the image by thresholding colours in HSV space, the second filters according to gradients in HLS space. Finally these two tracks are combined together at the end. This makes the edge detection pipeline robust to many types of road conditions and potential occlusion in the image such as shadows and adverse lighting conditions.

### Track 1: Colour Mask Filtering

The image is first converted into HSV space using the OpenCV function `cv2.cvtColor(image, cv2.COlOR_RGB2HSV)`.

Any image can then be filtered by colour according to a lower and upper threshold in HSV space

```python
def colour_thresh(image, thesh_min=np.array([0,0,0]), thresh_max=np.array([255,255,255])):
colour_mask = cv2.inRange(image, thresh_min, thresh_max)/255
return colour_mask
```

Original unprocessed image

![Original Image](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/edge_detect_orig.png?raw=true)

![White](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/yellow_filter.png?raw=true)![]()https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/yellow_filter.png?raw=true)
### Track 2: Gradient Filtering
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTYyNzkxMDI4NF19
-->
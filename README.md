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

To identify the pixels belonging to the lane lines in the image, combinations of gradient thresholding and colour space threshholding techniques are used. All of the routines and the code associated with the edge dection processing  can be found in the `edge_detection.py` file.

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


The image is then filtered by a white and yellow colour mask in HSV space according to the following threshold parameters and produces the following binary masks

```python
#Set parametrs for yellow filtering in HSV
    hsv_yellow_low = np.array([0, 100, 100])
    hsv_yellow_high = np.array([50, 255, 255])

    #Set parameters for white filtering in HSV
    hsv_white_low = np.array([20, 0, 180])
    hsv_white_high = np.array([255, 80, 255])
```

![White](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/yellow_filter.png?raw=true)![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/white_filter.png?raw=true)

### Track 2: Gradient Filtering

The image is additionally converted into HLS space using the OpenCV function `cv2.cvtColor(image, cv2.COlOR_RGB2HLS)`.

The gradients along the x and y components of the image are computed on both the L and S channels. The image gradients can be computed in OpenCV using the `cv2.sobel()` function. The gradients are computed in the code using the following function

```python
#Calculate the directional gradient
def abs_sobel_thresh(img_channel, orient='x', sobel_kernel=3 , thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    binary_output = np.uint8(255*abs_sobel/np.max(abs_sobel))
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask

```

Computing the x and y gradients for the S and L image channels, thresholding them according the the following parameters, and then combining the result via a bit-wise or produces the following binary masks.

```python
 img_grad_x = abs_sobel_thresh(img, 'x', 5, 50, 255)
 img_grad_y = abs_sobel_thresh(img, 'y', 5, 50, 255)
```
![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/gradient_l_mask.png?raw=true)![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/gradient_s_mask.png?raw=true)


### Combining tracks

Finally the binary masks from the two tracks are combined together to produce the final binary image mask detecting the road lane lines. Its useful to visualise the binary mask in RBG space where the colour mask is plotting on the green channel and the gradient mask on the blue. (note: in reality the algorithm only uses a single channel binary mask, this is just a useful visualisation technique)

![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/final_binary_mask.png?raw=true)


## Perspective Transformation

To identify and fit polynomials lines curving lanes it is useful to transform the perspective of the front facing camera to an overhead or 'birdseye' view of the lane. This is done using the `cv2.getPerspectiveTranform()` function and finding a set of source and destination points to warp the image to perform the perspective transform.


| Source     | Destination      | 
| ------------- |:-------------:| 
| (220, 720)    | (320, 720) | 
| (1110, 720)      | (920, 720)     |   
| (570, 470) | (320,1 )     |   
|  (722, 470)  |    (920, 1)  |


The code to implement the perspective transform is contained within the `PerspectiveTransform ` class in `perspective_transform.py`.

![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/original_src.png?raw=true)![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/perspective_trans.png?raw=true)


## Finding the lines

Having identified all possible pixels that could potentially be associated with lane lines in this image and transformed into a overhead perspective, the task is now to explicitly decide which pixels do belong to a lane line and which lane line they belong to. 

To do this we must first blindly search the image using a sliding window method to find the left and right line pixels. For video data streams the left and right lane pixels can be found by searching in the region close to lines identified in previous frames (this is a valid approach assuming the changes in the lane properties between video frames is small).

### Sliding Window Search:

To find the all the pixels associated with each lane line, a series of sliding windows are transversed along the height of the image so that it captures all of the pixels belonging to the lane within the windows. Each window is represented by the `Window` class found in `window.py`. The left and right lane windows are initialised in top and bottom y-position upon construction of the `LaneDectection` object via the following initialisation routine.

```python
 def initalise_windows(self):
        leftx = np.int(self.w/4)
        rightx = np.int(3*(self.w/4))   
        for i in range(self.nwindows)
            left_window = Window(self.h - (i+1)*self.window_height, 
                                 self.h - i*self.window_height, 
                                 leftx, self.margin, self.tol)         
            right_window = Window(self.h - (i+1)*self.window_height, 
                                 self.h - i*self.window_height, 
                                 rightx, self.margin, self.tol)
            
            self.left_window.append(left_window)
            self.right_window.append(right_window)
```

In order to fit these windows to the correct x-positions in the image, we first take a histogram of pixel count in the lower half of the binary image mask against image column. This should give two peaks in the pixel count histogram that can give the x position to fit the first window of each line.

From the first window, a sliding window placed along the centres of each line following up to the top of the frame. The full implementation of this is found in the ` fit_windows(self, image)` method within the `LaneDetection` class. The result of fitting a sliding window search to the binary image mask is shown below.

![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/Sliding_window.png?raw=true)


### Previous Region Search:

If the lane lines have been detected in previous video frames, searching the image blindly is an unnecessary computational expense. It is instead more effective to search the region around the lines identified in the previous frame or series of frames. The implementation of this previous region search 

![](https://github.com/joshwadd/Advanced_Lane_Detection/blob/master/output_images/regionsearch.png?raw=true)

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQ1MDUwMjEzXX0=
-->
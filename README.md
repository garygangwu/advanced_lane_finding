# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### With the driving video as an input, the goal is to effectively identify lane boundaries and present numerical estimation of lane curvature and vehicle position.

Final Videos
| <a href="http://www.youtube.com/watch?feature=player_embedded&v=4ly1T2cCRLU" target="_blank"><img src="http://img.youtube.com/vi/4ly1T2cCRLU/0.jpg" alt="Project Video" width="240" height="180" border="10" /></a> | <a href="http://www.youtube.com/watch?feature=player_embedded&v=33gyWJF9V1k" target="_blank"><img src="http://img.youtube.com/vi/33gyWJF9V1k/0.jpg" alt="Road Track One" width="240" height="180" border="10" /></a> |
|---|---|
|[Project Video (easy one) - Youtube](https://youtu.be/4ly1T2cCRLU)|[Road Track two (hard one) - Youtube](https://youtu.be/33gyWJF9V1k) |

---

The steps to execute this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

## Camera Calibration

A chessboard is used to do the camera calibration, because its regular high contrast pattern makes it easy to detect shape automatically.

| Original image | Undistorted image |
|:---:|:---:|
|<img src="camera_cal/calibration1.jpg" width="360" alt="original Image" /> |<img src="camera_cal/calibration1_undistorted.jpg" width="360" alt="undistorted Image" />|

First, a chessboard image is converted to a grayscale image. Then the OpenCV function `findChessboardCorners()` and `drawChessboardCorners()` can automatically find and draw internal corners in an image of a chessboard pattern. These internal corners of the chessboard along with their pixel ocations are fed into `cv2.calibrateCamera()`, which returns camera calibration and distortion coefficients. These can then be used as input parameters to the `cv2.undistort()` function to undo the effects of distortion on any image produced by the same camera.

##### More examples

|<img src="camera_cal/calibration2_recovered.jpg" width="480" alt="original Image" />|<img src="camera_cal/calibration3_recovered.jpg" width="480" alt="original Image" />|
|---|---|
|<img src="camera_cal/calibration4_recovered.jpg" width="480" alt="original Image" />|<img src="camera_cal/calibration5_recovered.jpg" width="480" alt="original Image" />|
|<img src="camera_cal/calibration6_recovered.jpg" width="480" alt="original Image" />|<img src="camera_cal/calibration7_recovered.jpg" width="480" alt="original Image" />|

## Pipeline (single images)

### 1, Image distortion correction

With the assumption that the distortion coefficients from the same camera are consistent, I used the same calibration parameters to undistort the images during driving. The image perspective transformation code is in [perspective_transform_utils.py](https://github.com/garygangwu/advanced_lane_finding/blob/master/perspective_transform_utils.py)

| Original image | Undistorted image |
|:---:|:---:|
|<img src="test_images/test5.jpg" width="360" alt="original Image" /> |<img src="test_images/test5_undistorted.jpg" width="360" alt="undistorted Image" />|

### 2. Color and gradients transforms

I developed a set of color and gradients transform utility functions in [image_transform_utils.py](https://github.com/garygangwu/advanced_lane_finding/blob/master/image_transform_utils.py) to speed up my experiments to find proper thresholds and method combination to detect lanes while minimzing the road noices 

### 3. Perspective transform

### 4. Identify lane-line pixels and match their positions with a polynomial

### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center

### 6. Plot the results back to front-facing image

## Files and usage
* `camera_calibration_utils.py`: utility functions to do the camera calibration from the chessboard images

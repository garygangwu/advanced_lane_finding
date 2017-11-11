import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

AVAILABLE_PATTEN_SIZES = [
  (9, 5), (8, 6), (9, 6), (8, 5), (7, 3), (9, 4), (9, 3)
]

def get_chess_calibration_params(chess_img, nx, ny):
  # Init obj / img points
  objpoints = [] # 3D points in real world space
  imgpoints = [] # 2D points in image plane
  objp = np.zeros((nx*ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

  # Convert to grayscale
  gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
  # Find the chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
  if ret == False:
    raise Exception('Failled in finding chess board corners') 

  # Add object points, image points
  imgpoints.append(corners)
  objpoints.append(objp)
  # Draw and display the corners
  cv2.drawChessboardCorners(chess_img, (nx, ny), corners, ret) 
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  return {
    'mtx': mtx,
    'dist': dist,
    'rvecs': rvecs,
    'tvecs': tvecs
  }


CARLIBRATION_PARAMS = get_chess_calibration_params(
  mpimg.imread('camera_cal/calibration1.jpg'),
  9,
  5)

def recover_distorted_image(img, calibration_params = CARLIBRATION_PARAMS):
  mtx = calibration_params['mtx']
  dist = calibration_params['dist']
  return cv2.undistort(img, mtx, dist, None, mtx)


def get_perspective_transform_image(undistorted_image):
  gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
  img_size = (gray.shape[1], gray.shape[0]) 

  ret = False
  for pattern in AVAILABLE_PATTEN_SIZES:
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if ret:
      break
  if ret == False:
    raise Exception('Failled in finding chess board corners')
  nx, ny = pattern[0], pattern[1]
  upper_left = corners[0]
  upper_right = corners[nx-1]
  lower_left = corners[-nx]
  lower_right = corners[-1]

  cv2.drawChessboardCorners(undistorted_image, (nx, ny), corners, ret)
  src = np.float32([ upper_left, upper_right, lower_left, lower_right ])
  offset = 100 # offset for dst points
  width = img_size[0] - offset
  length = img_size[1] - offset
  dst = np.float32([ [offset,offset], [width,offset], [offset,length], [width,length] ])
  M = cv2.getPerspectiveTransform(src, dst)
  return cv2.warpPerspective(undistorted_image, M, img_size)


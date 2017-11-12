import matplotlib.image as mpimg
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sobel_xy(img, orient='x', sobel_kernel=3, threshold=(35, 100)):
  """
  Apply the Sobel operator on either X or Y direction
  """
  if orient == 'x':
      abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
  if orient == 'y':
      abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
  scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
  binary_output = np.zeros_like(img)
  binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
  return binary_output

def mag_thresh(img, sobel_kernel=3, threshold=(30, 255)):
  """
  Return the magnitude of the gradient
  """
  sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  gradmag = np.sqrt(sobelx**2 + sobely**2)
  scale_factor = np.max(gradmag)/255
  gradmag = (gradmag/scale_factor).astype(np.uint8)
  binary_output = np.zeros_like(gradmag)
  binary_output[(gradmag >= threshold[0]) & (gradmag <= threshold[1])] = 1
  return binary_output


def dir_thresh(img, sobel_kernel=15, threshold=(0.7, 1.3)):
  """
  Computes the direction of the gradient
  """
  sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
  binary_output = np.zeros_like(absgraddir)
  binary_output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1
  return binary_output


def region_of_interest(img, vertices):
  """
  Applies an image mask.

  Only keeps the region of the image defined by the polygon
  formed from `vertices`. The rest of the image is set to black.
  """
  #defining a blank mask to start with
  mask = np.zeros_like(img)

  #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
  else:
    ignore_mask_color = 255

  #filling pixels inside the polygon defined by "vertices" with the fill color
  cv2.fillPoly(mask, vertices, ignore_mask_color)

  #returning the image only where mask pixels are nonzero
  masked_image = cv2.bitwise_and(img, mask)
  return masked_image


def gradient_combine(img, th_x, th_y, th_mag, th_dir):
  """
  Find lane lines with gradient information of Red channel
  """
  sobelx = sobel_xy(img, orient='x', threshold=th_x)
  sobely = sobel_xy(img, orient='y', threshold=th_y)
  mag_img = mag_thresh(img, sobel_kernel=3, threshold=th_mag)
  dir_img = dir_thresh(img, sobel_kernel=15, threshold=th_dir)

  gradient_comb = np.zeros_like(img)
  gradient_comb[
    ((sobelx != 0) & (mag_img != 0) & (dir_img != 0)) #|
    #((sobelx != 0) & (sobely != 0))
  ] = 1

  return gradient_comb


def filter_colors_hls(rgb_img, keep_yellow = True, keep_white = True):
  """
  Only keep the white and yellow color in HLS color space
  """
  converted_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
  mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

  if keep_yellow:
    yellow_dark = np.array([ 20, 120, 30])
    yellow_light = np.array([ 40,255,255])
    yellow_mask = cv2.inRange(converted_img, yellow_dark, yellow_light)
    mask = mask | yellow_mask

  if keep_white:
    white_dark = np.array([  0, 200, 0])
    white_light = np.array([255,255,255])
    white_mask = cv2.inRange(converted_img, white_dark, white_light)
    mask = mask | white_mask

  return cv2.bitwise_and(converted_img, converted_img, mask = mask)


def filter_colors_hsv(rgb_img, keep_yellow = True, keep_white = True):
  """
  Only keep the white and yellow color in HSV color space
  """
  converted_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
  mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

  if keep_yellow:
    yellow_dark = np.array([15, 127, 127], dtype=np.uint8)
    yellow_light = np.array([25, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(converted_img, yellow_dark, yellow_light)
    mask = mask | yellow_mask

  if keep_white:
    white_dark = np.array([0, 0, 200], dtype=np.uint8)
    white_light = np.array([255, 30, 255], dtype=np.uint8)
    white_mask = cv2.inRange(converted_img, white_dark, white_light)
    mask = mask | white_mask

  return cv2.bitwise_and(converted_img, converted_img, mask=mask)


def filter_colors_rgb(rgb_img, keep_yellow = True, keep_white = True):
  """
  Only keep the white and yellow color in RGB color space
  """
  mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

  if keep_yellow:
    yellow_dark = np.array([180, 180, 0], dtype=np.uint8)
    yellow_light = np.array([255, 255, 170], dtype=np.uint8)
    yellow_mask = cv2.inRange(rgb_img, yellow_dark, yellow_light)
    mask = mask | yellow_mask

  if keep_white:
    white_dark = np.array([100, 100, 200], dtype=np.uint8)
    white_light = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(rgb_img, white_dark, white_light)
    mask = mask | white_mask

  return cv2.bitwise_and(rgb_img, rgb_img, mask=mask)

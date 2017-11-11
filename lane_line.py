import numpy as np
import cv2
import matplotlib.image as mpimg

class Line():
  def __init__(self):
    # x values of the last n fits of the line
    self.recent_xfitted = []
    #average x values of the fitted line over the last n iterations
    self.bestx = None
    #polynomial coefficients averaged over the last n iterations
    self.best_fit = None
    #polynomial coefficients for the most recent fit
    self.current_fit = np.array([])
    #radius of curvature of the line in some units
    self.radius_of_curvature = None
    #distance in meters of vehicle center from the line
    self.line_base_pos = None
    #difference in fit coefficients between last and new fits
    self.diffs = np.array([0,0,0], dtype='float')
    #x values for detected line pixels
    self.allx = np.array([])
    #y values for detected line pixels
    self.ally = np.array([])
    #windwos to capture pixels
    self.windows = []
    # All the raw pixels found to construct the curve
    self.raw_x = None
    self.raw_y = None


  def is_line_detected(self):
    return self.current_fit.any()


# Left and right lane instance
left_lane = Line()
right_lane = Line()

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30.0/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def detect_raw_pixels_from_scratch(binary_warped):
  # Assuming you have created a warped binary image called "binary_warped"
  # Take a histogram of the bottom half of the image
  histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

  # Find the peak of the left and right halves of the histogram
  # These will be the starting point for the left and right lines
  midpoint = np.int(histogram.shape[0]/2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint

  # Choose the number of sliding windows
  nwindows = 9

  # Set height of windows
  window_height = np.int(binary_warped.shape[0]/nwindows)
  # Identify the x and y positions of all nonzero pixels in the image
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])

  # Current positions to be updated for each window
  leftx_current = leftx_base
  rightx_current = rightx_base
  # Set the width of the windows +/- margin
  margin = 100
  # Set minimum number of pixels found to recenter window
  minpix = 50
  # Create empty lists to receive left and right lane pixel indices
  left_lane_inds = []
  right_lane_inds = []

  left_windows = []
  right_windows = []

  # Step through the windows one by one
  for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    left_windows.append(((win_xleft_low, win_y_low), (win_xleft_high,  win_y_high)))
    right_windows.append(((win_xright_low, win_y_low), (win_xright_high,  win_y_high)))

    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
      leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
      rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  # Concatenate the arrays of indices
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]
  return leftx, lefty, rightx, righty, left_windows, right_windows


def compute_line(lane, binary_warped):
  pixel_y = np.concatenate((lane.raw_y, lane.ally))
  pixel_x = np.concatenate((lane.raw_x, lane.allx))

  new_fit = np.polyfit(pixel_y, pixel_x, 2)

  ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0], dtype=np.int32)
  fitx = (new_fit[0]*ploty**2 + new_fit[1]*ploty + new_fit[2]).astype(np.int32)

  lane.current_fit = new_fit
  lane.ally = ploty
  lane.allx = fitx


def draw_image_with_lanes(left_lane, right_lane, binary_warped):
  out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
  if not left_lane.is_line_detected() or not right_lane.is_line_detected():
    return out_img

  img_shape = out_img.shape
  line_color = [0, 255, 0]
  line_width = 10
  nx = out_img.shape[1]
  for i in range(len(left_lane.ally)):
    y = left_lane.ally[i]
    left_x = left_lane.allx[i]
    right_x = right_lane.allx[i]
    for x in range(nx):
      if x >= left_x and x <= right_x and x >= 0 and x < 1280:
        out_img[y][x] = [0, 0, 100]
    for w in range(line_width):
      if left_x + w >= 0 and left_x + w < 1280:
        out_img[y][left_x+w] = line_color
      if right_x - w >= 0 and right_x - w < 1280:
        out_img[y][right_x-w] = line_color
  return out_img


def draw_debug_image(left_lane, right_lane, binary_warped):
  out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
  out_img[left_lane.raw_y, left_lane.raw_x] = [255, 255, 255]
  out_img[right_lane.raw_y, right_lane.raw_x] = [255, 255, 255]
  line_width = 3
  if left_lane.is_line_detected():
    for i in range(len(left_lane.ally)):
      y = left_lane.ally[i]
      x = left_lane.allx[i]
      for w in range(line_width):
        if x + w >= 0 and x + w < 1280:
          out_img[y, x+w] = [255, 0, 0]
  if right_lane.is_line_detected():
    for i in range(len(right_lane.ally)):
      y = right_lane.ally[i]
      x = right_lane.allx[i]
      for w in range(line_width):
        if x - w >= 0 and x - w < 1280:
          out_img[y, x-w] = [255, 0, 0]
  return out_img


def detect_raw_pixels_from_previous_lines(binary_warped, left_lane, right_lane):
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  margin = 100

  left_fit = left_lane.current_fit
  left_lane_inds = (
    (nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
    (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))
  )

  right_fit = right_lane.current_fit
  right_lane_inds = (
    (nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
    (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))
  )

  # Again, extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  return leftx, lefty, rightx, righty


def get_rad_of_curvature_status(left_lane, right_lane):
  ploty = left_lane.ally
  leftx = left_lane.allx
  rightx = right_lane.allx

  # Fit new polynomials to x,y in world space
  left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
  # Calculate the new radii of curvature
  y_eval = np.max(ploty)
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
  # Now our radius of curvature is in meters
  left_lane.radius_of_curvature = left_curverad
  right_lane.radius_of_curvature = right_curverad

  return 'Radius of curvature: {}m'.format(int(left_curverad + right_curverad / 2))


def get_driving_deviation_status(left_lane, right_lane):
  left_startx = left_lane.allx[-1]
  right_startx = right_lane.allx[-1]

  center_lane = (right_startx + left_startx) / 2
  lane_width = right_startx - left_startx

  center_car = 1280 / 2

  distance = abs(center_lane - center_car) * xm_per_pix
  if center_lane > center_car:
    deviation = '{0:.2f}m left of center'.format(distance)
  elif center_lane < center_car:
    deviation = '{0:.2f}m right of center'.format(distance)
  else:
    deviation = 'at the center of the lane'
  return deviation


def search_left_right_lines_from_binary_image(binary_warped):
  global left_lane
  global right_lane

  if not left_lane.is_line_detected() or not right_lane.is_line_detected():
    leftx, lefty, rightx, righty, left_windows, right_windows = \
      detect_raw_pixels_from_scratch(binary_warped)
    left_lane.windows = left_windows
    right_lane.windows = right_windows
  else:
    leftx, lefty, rightx, righty = \
      detect_raw_pixels_from_previous_lines(binary_warped, left_lane, right_lane)

  left_lane.raw_x = leftx
  left_lane.raw_y = lefty
  right_lane.raw_x = rightx
  right_lane.raw_y = righty

  compute_line(left_lane, binary_warped)
  compute_line(right_lane, binary_warped)

  curvature_status = get_rad_of_curvature_status(left_lane, right_lane)
  deviation_status = get_driving_deviation_status(left_lane, right_lane)
  debug_image = draw_debug_image(left_lane, right_lane, binary_warped)
  lane_image = draw_image_with_lanes(left_lane, right_lane, binary_warped)
  return lane_image, debug_image, curvature_status, deviation_status


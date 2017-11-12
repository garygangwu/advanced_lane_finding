from camera_calibration_utils import *
from image_transform_utils import *
from perspective_transform_utils import *
from lane_line import *
from moviepy.editor import VideoFileClip

first = True

def convert_image_to_color_binary(image, threshold = (20, 255)):
  #hls_filtered_img = filter_colors_hls(image)
  hsv_filtered_img = filter_colors_hsv(image)
  #rgb_filtered_img = filter_colors_rgb(image)

  #hls_filtered_gray_img = cv2.cvtColor(hls_filtered_img, cv2.COLOR_BGR2GRAY)
  hsv_filtered_gray_img = cv2.cvtColor(hsv_filtered_img, cv2.COLOR_BGR2GRAY)
  #rgb_filtered_gray_img = cv2.cvtColor(rgb_filtered_img, cv2.COLOR_BGR2GRAY)
  #combined_filtered_gray_img = hls_filtered_gray_img | hsv_filtered_gray_img #| rgb_filtered_gray_img
  combined_filtered_gray_img = hsv_filtered_gray_img

  thresh_min = threshold[0]
  thresh_max = threshold[1]
  binary = np.zeros_like(combined_filtered_gray_img)
  binary[
    (combined_filtered_gray_img >= thresh_min) &
    (combined_filtered_gray_img <= thresh_max)
  ] = 1

  #plt.imsave('output_videos/2_hls_filtered_gray.jpg', hls_filtered_gray_img)
  #plt.imsave('output_videos/2_hsv_filtered_gray.jpg', hsv_filtered_gray_img)
  #plt.imsave('output_videos/2_rgb_filtered_gray.jpg', rgb_filtered_gray_img)
  #plt.imsave('output_videos/2_combined_filtered_gray.jpg', combined_filtered_gray_img)
  return binary

def convert_image_to_gradient_binary(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  sobel_x = sobel_xy(gray)
  mag_img = mag_thresh(gray)
  dir_img = dir_thresh(gray)

  gradient_comb = np.zeros_like(gray)
  gradient_comb[
    ((sobel_x != 0) & (mag_img != 0) & (dir_img != 0)) & (gray > 140)
  ] = 1

  imshape = gray.shape
  vertices = np.array(
    [[(50,imshape[0]),
      (imshape[1]/2-50, imshape[0] * 0.6),
      (imshape[1]/2+50, imshape[0] * 0.6),
      (imshape[1]-50,imshape[0])]],
    dtype=np.int32)

  masked_edges = region_of_interest(gradient_comb, vertices)

  plt.imsave('output_videos/3_sobel_x.jpg', sobel_x * 255)
  plt.imsave('output_videos/3_mag_img.jpg', mag_img * 255)
  plt.imsave('output_videos/3_dir_img.jpg', dir_img * 255)
  plt.imsave('output_videos/3_gradient_comb.jpg', gradient_comb * 255)
  plt.imsave('output_videos/3_masked_edges.jpg', masked_edges * 255)
  return masked_edges


def concatenate_images(
      orginal_image, undistorted_image, overlay_image,
      color_binary, bird_view_debug_image, bird_view_lane_image):
  plt.imsave('output_videos/1_orginal_image.jpg', orginal_image)
  plt.imsave('output_videos/6_output_debug_image.jpg', bird_view_debug_image)
  plt.imsave('output_videos/6_output_lane_image.jpg', bird_view_lane_image)
  plt.imsave('output_videos/6_overlay_image.jpg', overlay_image)

  gradient_binary = mag_thresh(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY))

  small_size_shape = (orginal_image.shape[1] / 2, orginal_image.shape[0] / 2)

  cololr_filtered_image = np.dstack((color_binary, color_binary, color_binary)) * 255
  cv2.putText(cololr_filtered_image, 'HLS/HSV Color Transform', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 3)
  resized_cololr_filtered_image = cv2.resize(cololr_filtered_image, small_size_shape)

  gradient_filtered_image = np.dstack((gradient_binary * 255, gradient_binary * 255, gradient_binary * 0))
  cv2.putText(gradient_filtered_image, 'Gradient Transform', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 3)
  resized_gradient_filtered_image = cv2.resize(gradient_filtered_image, small_size_shape)

  bird_view_image = get_bird_view_transformed_image(undistorted_image)
  cv2.putText(bird_view_image, 'Bird view', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 3)
  resized_bird_view_image = cv2.resize(bird_view_image, small_size_shape)

  resized_bird_view_debug_image = cv2.resize(bird_view_debug_image, small_size_shape)
  resized_bird_view_lane_image = cv2.resize(bird_view_lane_image, small_size_shape)

  resized_orginal_image = cv2.resize(orginal_image, small_size_shape)
  cv2.putText(resized_orginal_image, 'Original Video', (400,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

  left_side_image = np.concatenate((resized_orginal_image, resized_gradient_filtered_image), axis=0)
  upper_side_image = np.concatenate((overlay_image, left_side_image), axis=1)
  bottom_side_image = np.concatenate((resized_bird_view_image, resized_bird_view_debug_image), axis=1)
  bottom_side_image = np.concatenate((bottom_side_image, resized_cololr_filtered_image), axis=1)
  output_image = np.concatenate((upper_side_image, bottom_side_image), axis=0)

  plt.imsave('output_videos/6_output_image.jpg', output_image)
  return output_image


def process_image(orginal_image):
  undistorted_image = recover_distorted_image(orginal_image)
  color_binary = convert_image_to_color_binary(undistorted_image)
  #gradient_binary = convert_image_to_gradient_binary(undistorted_image)
  comb_binary = color_binary #| gradient_binary

  bird_view_binary = get_bird_view_transformed_image(comb_binary)
  bird_view_lane_image, bird_view_debug_image, curvature_status, deviation_status= \
    search_left_right_lines_from_binary_image(bird_view_binary)

  front_view_lane_img = get_front_view_transformed_image(bird_view_lane_image)
  overlay_image = cv2.addWeighted(undistorted_image, 1, front_view_lane_img, 1, 0)

  cv2.putText(overlay_image, curvature_status, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)
  cv2.putText(overlay_image, deviation_status, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)

  return concatenate_images(
    orginal_image, undistorted_image, overlay_image,
    color_binary, bird_view_debug_image, bird_view_lane_image)


video_file_names = [
  'challenge_video.mp4'
#  'project_video.mp4',
#  'harder_challenge_video.mp4'
]

for video_file_name in video_file_names:
  clip = VideoFileClip(video_file_name)
  new_clip = clip.fl_image(process_image)
  output_file = 'output_videos/' + video_file_name
  new_clip.write_videofile(output_file, audio=False)

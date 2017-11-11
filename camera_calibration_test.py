from camera_calibration_utils import *

for i in range(1, 20):
  fname = 'camera_cal/calibration' + str(i) + '.jpg'
  print fname
  img = mpimg.imread(fname)
  undistorted = recover_distorted_image(img)
  warped_image = get_perspective_transform_image(undistorted)
  f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
  f.tight_layout()
  ax1.imshow(img)
  ax1.set_title('Original Image', fontsize=20)
  ax2.imshow(undistorted)
  ax2.set_title('Undistored Image', fontsize=20)
  ax3.imshow(warped_image)
  ax3.set_title('Perspective Image', fontsize=20)
  plt.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.05)
  plt.savefig('camera_cal/calibration' + str(i) + '_recovered.jpg')

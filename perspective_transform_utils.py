import cv2
import numpy as np

src = np.float32(np.array([[550, 477], [160, 720], [1120, 720], [730, 477]]))
dst = np.float32(np.array([[320, 0], [320, 720], [960, 720], [960, 0]]))

M = cv2.getPerspectiveTransform(src, dst)
M_inverse = cv2.getPerspectiveTransform(dst, src)

def get_bird_view_transformed_image(image):
  img_size = (image.shape[1], image.shape[0])
  return cv2.warpPerspective(image, M, img_size)

def get_front_view_transformed_image(bird_view_image):
  img_size = (bird_view_image.shape[1], bird_view_image.shape[0])
  return cv2.warpPerspective(bird_view_image, M_inverse, img_size)

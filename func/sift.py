"""
SIFT descriptor
"""

import cv2
import numpy as np

def SIFTdesc(image, select_kp=False, kp_number=50):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if select_kp:
		sift = cv2.xfeatures2d.SIFT_create(kp_number)
	else:
		sift = cv2.xfeatures2d.SIFT_create()

	# Find keypoints and descriptors
	kp, desc = sift.detectAndCompute(gray, None)
	print("# descriptors: {}, type of data: {}, shape: {}".format(len(desc), type(desc), desc.shape))

	cv2.drawKeypoints(gray, kp, image, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	return image
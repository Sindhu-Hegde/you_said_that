import dlib, cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_max_area(boxes):
	# get face box with max area
	max_area, max_box = -np.inf, None
	for b in boxes:
		w = b.rect.right() - b.rect.left()
		h = b.rect.bottom() - b.rect.top()
		
		area = w * h
		if area > max_area:
			max_area = area
			max_box = b
	return max_box	


# Function to plot two images
def plotImages(im1, im2, t1, t2):
	
	fig=plt.figure(figsize=(15, 15))
	plt.subplot(1, 2, 1)
	plt.title(t1)
	plt.xticks([])
	plt.yticks([])
	if (im1.ndim == 2):
		plt.imshow(im1, cmap='gray') 
	elif (im1.ndim == 3):
		plt.imshow(im1) 

	plt.subplot(1, 2, 2)
	plt.title(t2)
	plt.xticks([])
	plt.yticks([])
	if (im1.ndim == 2):
		plt.imshow(im2, cmap='gray') 
	elif (im1.ndim == 3):
		plt.imshow(im2) 
	plt.show()
	plt.savefig("face.png")

	return


def crop_box(img, box):
	crop = img[max(box.rect.top(), 0) : box.rect.bottom(), max(box.rect.left(), 0) : box.rect.right()]
	return crop


def detect_faces(img_path):
	# reads the image if it is a string
	if type(img_path) == str:
		img = dlib.load_rgb_image(img_path)
	else:
		img = img_path

	detector = dlib.cnn_face_detection_model_v1('scripts/data/mmod_human_face_detector.dat')

	boxes = detector(img, 1)
	if len(boxes) == 0:
		print("The image doesn't have any face!!!")
		return None
	box = get_max_area(boxes)

	
	crop_img = crop_box(img, box)
	crop_img = cv2.resize(crop_img, (112, 112))

	return crop_img
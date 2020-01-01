import dlib, cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
if tf.test.gpu_device_name():
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
	print("Please install GPU version of TF")


detector, predictor = None, None 

def init(ckpt='scripts/data/shape_predictor_68_face_landmarks.dat'):
	global detector, predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(ckpt)

def get_max_area(boxes):
	# get face box with max area
	max_area, max_box = -np.inf, None
	for b in boxes:
		w = b.right() - b.left()
		h = b.bottom() - b.top()
		
		area = w * h
		if area > max_area:
			max_area = area
			max_box = b
	return max_box	

def crop_box(img, box):
	crop = img[max(box.top(), 0) : box.bottom(), max(box.left(), 0) : box.right()]
	return crop


def detect_face(img_path):

	# reads the image if it is a string
	if type(img_path) == str:
		img = dlib.load_rgb_image(img_path)
	else:
		img = img_path

	if not detector:
		init()

	boxes = detector(img, 0)
	if len(boxes) == 0:
		return None
	box = get_max_area(boxes)

	crop_img = crop_box(img, box)
	crop_img = cv2.resize(crop_img, (112, 112))

	return crop_img

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
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face related utils implemented here.')
	parser.add_argument('input', help='Input image')
	args = parser.parse_args()

	crop_img = detect_face(args.input)

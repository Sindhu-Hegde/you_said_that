import cv2
import numpy as np
import matplotlib.pyplot as plt

### Function to read the input image 
def read_image(filename, display=False):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if(display):
		print("Image dimension:", img.shape)

	return img
 
# Function to display the image
def write_image(img, fname="image"):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	cv2.imwrite(fname + ".png", img)
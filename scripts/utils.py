import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists, isdir, basename, join, splitext
from glob import glob

# Function to get all the folders in the given path
def get_folders(path):
	folder_paths = [files for files in glob(path + "/*") if isdir(files)]
	folders = [basename(folder_path) for folder_path in folder_paths]
	return folders


# Function to get all the files in the given folder path
def get_files(path, extension=[".mp4"]):
	all_files = []
	all_files.extend([join(path, basename(fname))
					for fname in glob(path + "/*")
					if splitext(fname)[-1].lower() in extension])
	return all_files

# Function to extract the frame number of the given input filename
def get_frame_num(fname):
	frame_num = fname.split("/")[-1].split(".")[0].split("_")[1]
	return int(frame_num)


# Function to read the input image 
def read_image(filename, display=False):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if(display):
		print("Image dimension:", img.shape)

	return img
 
# Function to write the image
def write_image(img, fname="image"):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	cv2.imwrite(fname + ".png", img)
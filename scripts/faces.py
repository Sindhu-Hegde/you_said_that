import cv2
import random
import argparse
from .face_utils import *
 
 
def extract_frames(filename):
	cap= cv2.VideoCapture(filename)

	frames = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		
		if ret == False:
			break
		
		frames.append(frame)
	
	cap.release() 
	cv2.destroyAllWindows()

	return frames

 
def extract_face(filename):

	frames = extract_frames(filename)

	faces = []
	for i in range(len(frames)):
		face = detect_face(frames[i])
		faces.append(face)

	return faces


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face related utils')
	parser.add_argument('input', help='Input image')
	args = parser.parse_args()

	crop_img = detect_face(args.input)

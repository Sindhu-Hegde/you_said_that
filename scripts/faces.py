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

	num_frames = len(frames)
	gt_frame = frames[num_frames//2]
	feature_frame = random.choice(frames[(num_frames//2)+1:])

	return gt_frame, feature_frame

 
def extract_face(filename):

	gt_frame, feature_frame = extract_frames(filename)

	gt_crop_img = detect_faces(gt_frame)
	feature_crop_img = detect_faces(feature_frame)

	return gt_crop_img, feature_crop_img


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face related utils implemented here.')
	parser.add_argument('input', help='Input image')
	args = parser.parse_args()

	crop_img = detect_faces(args.input)

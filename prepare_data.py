import argparse
import os
from os.path import exists, isdir, basename, join, splitext
from glob import glob
import pickle
import numpy as np
import cv2
from numpy import newaxis
from scripts.utils import *
from extract_features import get_folders, get_files
from sklearn.model_selection import train_test_split


def read_file(input_files):
	all_features_dict = {}
	gt_features = {}
	gt = None
	for i, fname in enumerate(input_files):
		if("_gt" not in fname):

			if(splitext(fname)[-1].lower() == ".pkl"):
				features = read_audio_features(fname)
			elif(splitext(fname)[-1].lower() == ".jpg"):
				f1 = fname
				f2 = fname.split(".")[0] + "_gt.jpg"
				if(os.path.exists(f1) and os.path.exists(f2)):
					features, gt = read_image_features(f1, f2)
				else:
					features = None
			else:
				features = None

			if(features is not None):
				all_features_dict[fname] = features
			if(gt is not None):
				gt_features[fname] = gt

	return all_features_dict, gt_features

def read_audio_features(filename):
		
	with open(filename, "rb") as fp:   
		features = pickle.load(fp)
	
	return features


def read_image_features(file1, file2):

	img_feature = read_image(file1)
	img_gt = read_image(file2)
	img_gt_masked = img_gt.copy()

	# Mask the lower part of the ground truth image
	lower_index = img_gt_masked.shape[0]//2
	upper_index = img_gt_masked.shape[0]
	img_gt_masked[lower_index:upper_index, :upper_index] = [0,0,0]	

	# Concatenate the feature and the ground truth images channel wise
	concatenated_feature = np.dstack((img_feature, img_gt_masked))
	
	return concatenated_feature, img_gt



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-f', '--feature_path', default='features', help='Feature path')	
	parser.add_argument('-p', '--processed_data_path', default='processed_data', required=False, \
						help='Path to store the processed data')
	args = parser.parse_args()


	print("---------------------")
	feature_path = args.feature_path
	folders = get_folders(feature_path)
	num_folders = len(folders)
	print("Total number of folders = ", num_folders)

	audio_features = {}
	identity_features = {}
	gt_features = {}

	for i, folder in enumerate(folders):
		print("Folder ", i+1, ":", folder)
		path = join(feature_path, folder)
		sub_folders = get_folders(path)
		for sub_folder in sub_folders:
			file_path = join(path, sub_folder)	

			audio_files = get_files(file_path, extension=[".pkl"])
			features1, _ = read_file(audio_files)
			audio_features.update(features1)

			image_files = get_files(file_path, extension=[".jpg"])
			features2, gt = read_file(image_files)
			if(features2 is not None):
				identity_features.update(features2)
				gt_features.update(gt)

	

	data_dir = args.processed_data_path
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	with open(data_dir + "/audio_features.pkl", "wb") as fp:   
	    pickle.dump(audio_features, fp)

	with open(data_dir + "/identity_features.pkl", "wb") as fp:   
	    pickle.dump(identity_features, fp)

	with open(data_dir + "/gt_features.pkl", "wb") as fp:   
	    pickle.dump(gt_features, fp)
 
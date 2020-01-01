import argparse
import os
from os.path import exists, isdir, basename, join, splitext
from glob import glob
import subprocess
import pickle
import cv2
from scripts.mfcc_features import *
from scripts.faces import *
from scripts.utils import *
import executor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

 
def save_audio_features(input_file, feature_dir, sampling_rate, n_mfcc, low_freq, high_freq, n_filt):

	# Create the directories
	main_dir = feature_dir + input_file.split("/")[-2]
	if not os.path.exists(main_dir):
		os.makedirs(main_dir)

	sub_dir = main_dir + "/" + input_file.split("/")[-1].split(".")[0]
	if not os.path.exists(sub_dir):
		os.makedirs(sub_dir)

		
	# Extract the MFCC features
	mfcc = extract_mfcc(input_file, sampling_rate, n_mfcc, low_freq, high_freq, n_filt)
	fname  = sub_dir + "/" + input_file.split("/")[-1].split(".")[0] + '.pkl'

	# Save the extracted features
	with open(fname, "wb") as file:
		pickle.dump(mfcc, file)


def save_identity_features(input_file, feature_dir):


	# Create the directories
	main_dir = feature_dir + input_file.split("/")[-2]
	if not os.path.exists(main_dir):
		os.makedirs(main_dir)

	sub_dir = main_dir + "/" + input_file.split("/")[-1].split(".")[0]
	if not os.path.exists(sub_dir):
		os.makedirs(sub_dir)


	# Extract the faces 
	faces = extract_face(input_file)

	for k in range(len(faces)):
		fname  = sub_dir + "/" + input_file.split("/")[-1].split(".")[0] + '_'+str(k+1)+'.jpg'
	
		# Save the faces
		if (faces[k] is not None):
			cv2.imwrite(fname, faces[k])



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d', '--data_path', help='Dataset path')	
	parser.add_argument('-w', '--num_workers', default=1, help='No of workers')
	parser.add_argument('-fd', '--feature_dir', default='features/', help='Path to save the features')
	parser.add_argument('-sr', '--sampling_rate', default=16000, required=False, \
						help='Sampling rate to extract MFCC features')
	parser.add_argument('-n_mfcc', '--no_of_mfcc_components', default=13, required=False, \
						help='No of cepstral coffecients to extract MFCC features')
	parser.add_argument('-lf', '--low_frequency', default=300, required=False, \
						help='Lower frequency component to extract MFCC features')
	parser.add_argument('-hf', '--high_frequency', default=3700, required=False, \
						help='Higher frequency component to extract MFCC features')
	parser.add_argument('-fc', '--filterbank_channels', default=40, required=False, \
						help='No of filterbank channels to extract MFCC features')
	args = parser.parse_args()


	print("---------------------")
	datasetpath = args.data_path
	folders = get_folders(datasetpath)
	num_folders = len(folders)
	print("Total number of folders = ", num_folders)


	files = []
	i=1
	for folder in folders:
		print("Folder: ", i)
		file_path = join(datasetpath, folder)
		file = get_files(file_path)
		files.extend(file)

		for f in file:
			save_audio_features(f, args.feature_dir, int(args.sampling_rate), \
								int(args.no_of_mfcc_components), int(args.low_frequency), \
								int(args.high_frequency), int(args.filterbank_channels))
	
		i+=1

	print("Successfully extracted and saved the audio features!!!")
	print("----------------------------------------------------------------------")

	jobs = [vid_file for vid_file in files]
	p = ThreadPoolExecutor(int(args.num_workers))
	futures = [p.submit(save_identity_features, j, args.feature_dir) for j in jobs]
	res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	print("Successfully extracted and saved the identity features!!!")
	print("----------------------------------------------------------------------")
	

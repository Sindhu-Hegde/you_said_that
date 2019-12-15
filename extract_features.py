import argparse
import os
from os.path import exists, isdir, basename, join, splitext
from glob import glob
import subprocess
import pickle
import cv2
from scripts.mfcc_features import *
from scripts.faces import *

def get_folders(datasetpath):
	folder_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
	folders = [basename(folder_path) for folder_path in folder_paths]
	return folders

def get_files(path, extension=[".mp4"]):
	all_files = []
	all_files.extend([join(path, basename(fname))
					for fname in glob(path + "/*")
					if splitext(fname)[-1].lower() in extension])
	return all_files

def save_features(input_file):

	duration = 0.0
	duration = subprocess.Popen("ffprobe -i %s -show_format -v quiet | sed -n 's/duration=//p'" % \
		(input_file) , stdout=subprocess.PIPE, shell=True)
	duration = float(duration.stdout.read())
	#print("Video duration: ", duration)

	main_dir = 'features/' + input_file.split("/")[-2]
	if not os.path.exists(main_dir):
		os.makedirs(main_dir)

	sub_dir = main_dir + "/" + input_file.split("/")[-1].split(".")[0]
	if not os.path.exists(sub_dir):
		os.makedirs(sub_dir)

		k=0
		i=0
		
		while True:
			
			mp4_file  = sub_dir + "/" + input_file.split("/")[-1].split(".")[0] + '_'+str(k)+'.mp4'
			subprocess.call('ffmpeg -i %s -ss %f -t 0.29 %s' % (input_file, i, mp4_file), shell=True)
			k+=1
			i+=0.24
			if(i+0.29 > duration):
				os.remove(mp4_file)
				break	

			# Extract the MFCC features
			mfcc = extract_mfcc(mp4_file)
			fname = sub_dir + "/" + mp4_file.split("/")[-1].split(".")[0] + ".pkl"
			with open(fname, "wb") as file:
				pickle.dump(mfcc, file)


			# Extract the faces 
			gt_frame, feature_frame = extract_face(mp4_file)
			fname_gt = sub_dir + "/" + mp4_file.split("/")[-1].split(".")[0] + "_gt.jpg"
			fname_feature = sub_dir + "/" + mp4_file.split("/")[-1].split(".")[0] + ".jpg"
			if(gt_frame is not None):
				cv2.imwrite(fname_gt, gt_frame)
			if(feature_frame is not None):
				cv2.imwrite(fname_feature, feature_frame)


			os.remove(mp4_file)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d', '--data_path', help='Dataset path')	
	args = parser.parse_args()


	print("---------------------")
	datasetpath = args.data_path
	folders = get_folders(datasetpath)
	num_folders = len(folders)
	print("Total number of folders = ", num_folders)


	for folder in folders:
		file_path = join(datasetpath, folder)
		files = get_files(file_path)
		for file in files:
			save_features(file) 
		
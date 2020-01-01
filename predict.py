from keras.models import load_model
import argparse
import pickle
import numpy as np
from scripts.utils import *
import subprocess
import os
from numpy import newaxis
from scripts.face_utils import *
from scripts.faces import *
from scripts.mfcc_features import *

   
def extract_test_features(test_file, test_img, feature_dir, fps=25):

	print("Test file: ", test_file)
	time = 0.0
	time = subprocess.Popen("ffprobe -hide_banner -loglevel panic -i %s -show_format -v quiet \
							| sed -n 's/duration=//p'" % (test_file) , stdout=subprocess.PIPE, \
							shell=True)
	time = float(time.stdout.read())
	print("Duration: ", time)
	num_frames = int(fps * time)

	test_dir = feature_dir + "/" + test_file.split("/")[-1].split(".")[0]
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	 
	# Extract the image features
	img = read_image(test_img)
	img_feature = detect_face(img)
	img_masked = img_feature.copy()
	lower_index = img_masked.shape[0]//2
	upper_index = img_masked.shape[0]
	img_masked[lower_index:upper_index, :upper_index] = [0,0,0]	
	concatenated_feature = np.dstack((img_feature, img_masked))

	#concatenated_feature = np.dstack((img_feature, img_feature, img_feature, img_feature, \
	#									img_feature, img_masked))

	

	# Extract theaudio features
	mfcc = extract_mfcc(test_file)
	
	fname  = test_dir + "/" + test_file.split("/")[-1].split(".")[0] + '.pkl'
	with open(fname, "wb") as file:
		pickle.dump(mfcc, file)

	audio_features = []
	identity_features = []
	
	for i in range(num_frames):
		time = (i+1) / fps
		start_time = time - 0.175
		if(start_time < 0):
			continue
		start_mfcc = int(start_time*100)
		feature = mfcc[:, start_mfcc:(start_mfcc + 35)]
		if(feature.shape != (12, 35)):
			continue

		audio_features.append(feature)
		identity_features.append(concatenated_feature)	    
		if((i+35) > mfcc.shape[1]):
			break
		
	x_test_audio = np.array(audio_features)
	x_test_audio = x_test_audio.astype('float32') 
	x_test_audio = x_test_audio[:, :, :, newaxis]
	print("Audio shape: ", x_test_audio.shape)

	x_test_identity = np.array(identity_features)
	x_test_identity = x_test_identity.astype('float32') / 255.
	print("Identity shape: ", x_test_identity.shape)
		

	return x_test_audio, x_test_identity


def predict(model, test_file, x_test_audio, x_test_identity, feature_dir):


	predicted_frames = model.predict([x_test_audio, x_test_identity])
	print("Predicted frames: ", predicted_frames.shape)

	test_dir = feature_dir + "/" + test_file.split("/")[-1].split(".")[0] + '/frames' 
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	predictions = []
	for i in range(len(predicted_frames)):
		im = np.clip(np.round(predicted_frames[i]*255), 0, 255)
		predictions.append(im)
		fname = test_dir + "/" + test_file.split("/")[-1].split(".")[0] + '_' + str(i+1)
		write_image(im, fname)

	return predictions


def generate_video(test_file, predictions, output_file_name, fps=25):

	height, width, layers = predictions[0].shape
	fname = 'output.avi' 
	video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
 
	for i in range(len(predictions)):
		img = cv2.cvtColor(predictions[i], cv2.COLOR_BGR2RGB)
		video.write(np.uint8(img))
	
	video.release()

	video_output = output_file_name + '.mkv'
	subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -i %s -c copy -map 0:v -map 1:a %s' % \
													(fname, test_file, video_output), shell=True) 


	os.remove("output.avi")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-m', '--model_path', help='Saved Model path')	
	parser.add_argument('-t', '--test_file', help='Test audio or video file')
	parser.add_argument('-f', '--test_frame', help='Test frame')
	parser.add_argument('-fd', '--test_feature_dir', default='test_features', required=False, \
						help='Path to dave test features')
	parser.add_argument('-o', '--output_file_name', default='output_video', required=False, \
						help='Name of the output video file')
	args = parser.parse_args()


	model = load_model(args.model_path)

	x_test_audio, x_test_identity = extract_test_features(args.test_file, args.test_frame, \
															args.test_feature_dir)

	predictions = predict(model, args.test_file, x_test_audio, x_test_identity, args.test_feature_dir)

	generate_video(args.test_file, predictions, args.output_file_name)

from keras.models import load_model
import argparse
import pickle
import numpy as np
from scripts.utils import *
import subprocess
import os
import librosa
from numpy import newaxis
from scripts.face_utils import *
from scripts.mfcc_features import *

def load_data(audio_fname, identity_fname, gt_fname):

	with open(audio_fname, "rb") as fp:   
	    x_test_audio = pickle.load(fp)

	with open(identity_fname, "rb") as fp:   
	    x_test_identity = pickle.load(fp)

	with open(gt_fname, "rb") as fp:   
	    y_test = pickle.load(fp)

	return x_test_audio, x_test_identity, y_test


def save_test_features(test_file, test_img):

	extension = test_file.split(".")[1]
	wav_file = test_file
	
	if (extension == 'mp4'):
		wav_file = test_file.split(".")[0] + ".wav"

		subprocess.call('ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s' % \
		(test_file, wav_file), shell=True)


	duration = 0.0
	duration = subprocess.Popen("ffprobe -i %s -show_format -v quiet | sed -n 's/duration=//p'" % \
		(wav_file) , stdout=subprocess.PIPE, shell=True)
	duration = float(duration.stdout.read())

	main_dir = 'test_features/' + wav_file.split("/")[-1].split(".")[0]
	if not os.path.exists(main_dir):
		os.makedirs(main_dir)

	k=0
	i=0

	audio_features = {}
	identity_features = {}
	while True:
		audio_file  = main_dir + "/" + wav_file.split("/")[-1].split(".")[0] + '_' + str(k) + '.wav'
		subprocess.call('ffmpeg -i %s -ss %f -t 0.32 %s' % (wav_file, i, audio_file), shell=True)
		k+=1
		i+=0.05
		if(i+0.32 > duration):
			os.remove(audio_file)
			break	

		# Extract the MFCC features
		mfcc = extract_mfcc(audio_file)
		fname = main_dir + "/" + audio_file.split("/")[-1].split(".")[0] + ".pkl"
		with open(fname, "wb") as file:
			pickle.dump(mfcc, file)
		audio_features[fname] = mfcc
		

		# Extract the image features
		img = read_image(test_img)
		img_feature = detect_faces(img)
		img_masked = img_feature.copy()
		lower_index = img_masked.shape[0]//2
		upper_index = img_masked.shape[0]
		img_masked[lower_index:upper_index, :upper_index] = [0,0,0]	
		concatenated_feature = np.dstack((img_feature, img_masked))
		fname = main_dir + "/" + audio_file.split("/")[-1].split(".")[0] + ".jpg"
		identity_features[fname] = concatenated_feature

		os.remove(audio_file)

	return wav_file, audio_features, identity_features


def load_test_features(audio_features, identity_features):

	audio_data = []
	for i in audio_features:
	    audio_data.append(audio_features[i])

	identity_data = []
	for i in identity_features:
	    identity_data.append(identity_features[i])

	x_test_audio = np.array(audio_data)
	x_test_audio = x_test_audio.astype('float32') 
	x_test_audio = x_test_audio[:, :, :, newaxis]
	print("Audio shape: ", x_test_audio.shape)

	x_test_identity = np.array(identity_data)
	x_test_identity = x_test_identity.astype('float32') / 255.
	print("Identity shape: ", x_test_identity.shape)

	return x_test_audio, x_test_identity



def predict(model, test_file, x_test_audio, x_test_identity):

	print ("X test audio: ", x_test_audio.shape)
	print ("X test identity: ", x_test_identity.shape)

	predicted_imgs = model.predict([x_test_audio, x_test_identity])

	main_dir = 'test_features/' + test_file.split("/")[-1].split(".")[0] + '/frames' 
	if not os.path.exists(main_dir):
		os.makedirs(main_dir)

	predictions = []
	for i in range(len(predicted_imgs)):
		im = np.clip(np.round(predicted_imgs[i]*255), 0, 255)
		predictions.append(im)
		fname = main_dir + "/" + test_file.split("/")[-1].split(".")[0] + '_' + str(i)
		write_image(im, fname)

	return predictions


def generate_video(audio_file, predictions):

	no_of_frames = len(predictions)
	time = 0.0
	time = subprocess.Popen("ffprobe -i %s -show_format -v quiet | sed -n 's/duration=//p'" % \
		(audio_file) , stdout=subprocess.PIPE, shell=True)
	time = float(time.stdout.read())
	fps = no_of_frames / time

	print("No of frames = ", no_of_frames)
	print("Time = ", time)
	print("FPS = ", fps)


	height, width, layers = predictions[0].shape
	fname = 'output.avi' 
	video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
 
	for i in range(len(predictions)):
		img = cv2.cvtColor(predictions[i], cv2.COLOR_BGR2RGB)
		video.write(np.uint8(img))
	
	video.release()

	video_dir = 'output_videos'
	if not os.path.exists(video_dir):
		os.makedirs(video_dir)

	video_output = video_dir + '/output.mkv'
	subprocess.call('ffmpeg -i %s -i %s -c copy -map 0:v -map 1:a %s' % \
													(fname, audio_file, video_output), shell=True) 


	os.remove(fname)
	


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-m', '--model_path', help='Saved Model path')	
	parser.add_argument('-t', '--test_file', help='Test audio or video file')
	parser.add_argument('-f', '--test_frame', help='Test frame')
	args = parser.parse_args()

	
	model = load_model(args.model_path)

	audio_file, audio_features, identity_features = save_test_features(args.test_file, args.test_frame)

	x_test_audio, x_test_identity = load_test_features(audio_features, identity_features)
	predictions = predict(model, args.test_file, x_test_audio, x_test_identity)

	generate_video(audio_file, predictions)

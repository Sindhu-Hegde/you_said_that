import librosa
import subprocess
import os

def extract_mfcc(input_file, sr=55000, n_mfcc=13):
	
	extension = input_file.split(".")[1]
	wav_file = input_file

	if(extension == ".mp4"):
		wav_file  = 'tmp.wav';

		subprocess.call('ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s' % \
			(input_file, wav_file), shell=True)

	
	x, sr = librosa.load(wav_file, sr=55000)

	mfccs = librosa.feature.mfcc(x, sr=sr, n_mfcc=13)
	mfccs = mfccs[1:]
	print(mfccs.shape)

	if(extension == ".mp4"):
		os.remove(wav_file)

	return mfccs

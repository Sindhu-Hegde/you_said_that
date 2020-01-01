import subprocess
import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav



def extract_mfcc(input_file, sr=16000, n_mfcc=13, lowfreq=300, highfreq=3700, nfilt=40):
	wav_file  = 'tmp.wav';

	subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
					-acodec pcm_s16le -ar 16000 %s' % (input_file, wav_file), shell=True)



	(rate,sig) = wav.read(wav_file)
	mfcc_feat = mfcc(sig,rate, lowfreq=lowfreq, highfreq=highfreq, nfilt=nfilt)
	mfccs = mfcc_feat[:,1:].T 
	#print(mfccs.shape)

	os.remove(wav_file)

	return mfccs

import numpy as np
import pickle
import os
import argparse
from numpy import newaxis
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, concatenate, \
Reshape, merge, BatchNormalization, Dropout
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from scripts.utils import *
from scripts.data_generator import *
from scripts.multi_gpu_model import *


def load_features(audio_fname, identity_fname, gt_fname):

	with open(audio_fname, "rb") as fp:   
		audio = pickle.load(fp)

	with open(identity_fname, "rb") as fp:   
		identity_features = pickle.load(fp)

	with open(gt_fname, "rb") as fp:   
		gt_features = pickle.load(fp)

	audio_features = {}
	for fname in identity_features:
		audio_fname = fname.split(".")[0] + '.pkl'
		audio_features[audio_fname] = audio[audio_fname]

	return audio_features, identity_features, gt_features



def model(audio_features, identity_features, gt_features, batch_size, steps_per_epoch, epochs, \
			workers, gpus):


	keys = list(audio_features.keys())
	
	keys_train, keys_val = train_test_split(np.array(keys), test_size=0.1)
	training_generator = DataGenerator(keys_train, audio_features, identity_features, gt_features, \
										batch_size)
	validation_generator = DataGenerator(keys_val, audio_features, identity_features, gt_features, \
										batch_size)
	print("Validation generator: ", validation_generator)
	

	# Audio encoder
	input_audio = Input(shape=(12, 35, 1)) 

	x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_audio)
	x = BatchNormalization()(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides = (1, 2), padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides = 2, padding='same')(x)
	x = BatchNormalization()(x)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.3)(x)
	encoded_audio = Dense(256, activation='relu')(x)


	# Identity encoder
	input_identity = Input(shape=(112, 112, 6)) 

	x = Conv2D(96, (7, 7), strides = 2, activation='relu', padding='same')(input_identity)
	x = BatchNormalization()(x)
	x_skip1 = MaxPooling2D((3, 3), strides=2, padding='same')(x)
	x_skip1 = BatchNormalization()(x_skip1)
	x_skip2 = Conv2D(256, (5, 5), strides=2, activation='relu', padding='same')(x_skip1)
	x_skip2 = BatchNormalization()(x_skip2)
	x_skip3 = MaxPooling2D((3, 3), strides=2, padding='same')(x_skip2)
	x_skip3 = BatchNormalization()(x_skip3)
	x = Conv2D(512, (3, 3), activation='relu', padding='same')(x_skip3)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.3)(x)
	encoded_identity = Dense(256, activation='relu')(x)

	concatenated_features = concatenate([encoded_audio, encoded_identity])

	# Decoder
	x = Dense(98, activation='relu')(concatenated_features)
	x = Dropout(0.3)(x)
	x = Reshape((7, 7, 2))(x)
	x = Conv2DTranspose(512, (6, 6), activation='relu', padding='same')(x)
	x = Conv2DTranspose(256, (5, 5), activation='relu', padding='same')(x)
	x = concatenate([x, x_skip3])
	x = Conv2DTranspose(96, (5, 5), strides=2, activation='relu', padding='same')(x)
	x = concatenate([x, x_skip2])
	x = Conv2DTranspose(96, (5, 5), strides=2, activation='relu', padding='same')(x)
	x = concatenate([x, x_skip1])
	x = Conv2DTranspose(64, (5, 5), strides=2, activation='relu', padding='same')(x)
	decoded = Conv2DTranspose(3, (5, 5), strides=2, activation='sigmoid', padding='same')(x)

	
	model = Model(inputs = [input_audio, input_identity], outputs = [decoded])

	try:
		model = ModelMGPU(model, gpus)
	except:
		pass

	optimizer = optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=False)
	model.compile(optimizer='adam', loss='mean_absolute_error')
	print(model.summary())

	model.fit_generator(generator=training_generator,
						validation_data=validation_generator,
						steps_per_epoch=steps_per_epoch,
						epochs=epochs,
						use_multiprocessing=True,
						workers=workers
					   )

	model_dir = 'saved_models'
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	model.save(model_dir + '/model.h5')



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-a', '--audio_train_file_path', help='Audio train file path')	
	parser.add_argument('-i', '--identity_train_file_path', help='Identity train file path')
	parser.add_argument('-gt', '--ground_truth_train_path', help='Ground truth train path')
	parser.add_argument('-se', '--steps_per_epoch', default=1000, required=False, \
						help='No of steps per epoch')
	parser.add_argument('-b', '--batch_size', default=64, required=False, help='Batch size')
	parser.add_argument('-e', '--epochs', default=20, required=False, help='No of epochs')
	parser.add_argument('-w', '--no_of_workers', default=1, required=False, help='No of workers')
	parser.add_argument('-gpu', '--no_of_gpus', default=1, required=False, help='No of GPUs')
	
	args = parser.parse_args()


	audio_features, identity_features, gt_features = load_features(args.audio_train_file_path, \
											args.identity_train_file_path, args.ground_truth_train_path)


	model(audio_features, identity_features, gt_features, int(args.batch_size), \
		int(args.steps_per_epoch), int(args.epochs), int(args.no_of_workers), int(args.no_of_gpus))

	
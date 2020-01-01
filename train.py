import numpy as np
import pickle
import argparse
from os.path import exists, isdir, basename, join, splitext
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, concatenate, \
Reshape, BatchNormalization, Dropout, Activation
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from extract_features import get_folders, get_files
from scripts.data_generator import *
from scripts.multi_gpu_model import *
from datetime import datetime
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def convolution(x, filters, kernel_size=3, strides=1, padding='same'):
	x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)
	x = Activation('relu')(x)
	return x

def transposed_convolution(x, filters, kernel_size=3, strides=1, padding='same'):
	x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, \
		padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)
	x = Activation('relu')(x)
	return x


def model(files, epochs, steps_per_epoch, gpus, num_still_images, model_dir):

	
	train_files, val_files = train_test_split(np.array(files), test_size=0.1)
	training_generator = DataGenerator(train_files, num_still_images)
	validation_generator = DataGenerator(val_files, num_still_images)


	# Audio encoder
	input_audio = Input(shape=(12, 35, 1)) 

	x = convolution(input_audio, 64, 3)
	x = convolution(x, 128, 3)
	x = MaxPooling2D((3, 3), strides = (1, 2), padding='same')(x)
	x = convolution(x, 256, 3)
	x = convolution(x, 256, 3)
	x = convolution(x, 512, 3)
	x = MaxPooling2D((3, 3), strides = 2, padding='same')(x)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	encoded_audio = Dense(256, activation='relu')(x)


	# Identity encoder
	dim = num_still_images*3 + 3
	input_identity = Input(shape=(112, 112, dim)) 
	
	x = convolution(input_identity, 96, 7, 2)
	x_skip1 = MaxPooling2D((3, 3), strides=2, padding='same')(x)
	x_skip2 = convolution(x_skip1, 256, 5, 2)
	x_skip3 = MaxPooling2D((3, 3), strides=2, padding='same')(x_skip2)
	x = convolution(x_skip3, 512, 3)
	x = convolution(x, 512, 3)
	x = convolution(x, 512, 3)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	encoded_identity = Dense(256, activation='relu')(x)

	# Concatenate the audio and identity features
	concatenated_features = concatenate([encoded_audio, encoded_identity])

	# Decoder
	x = Dense(98, activation='relu')(concatenated_features)
	x = Reshape((7, 7, 2))(x)
	x = transposed_convolution(x, 512, 6)
	x = transposed_convolution(x, 256, 5)
	x = concatenate([x, x_skip3])
	x = transposed_convolution(x, 96, 5, 2)
	x = concatenate([x, x_skip2])
	x = transposed_convolution(x, 96, 5, 2)
	x = concatenate([x, x_skip1])
	x = transposed_convolution(x, 64, 5, 2)
	decoded = Conv2DTranspose(3, (5, 5), strides=2, activation='sigmoid', padding='same')(x)

	model = Model(inputs = [input_audio, input_identity], outputs = [decoded])
 
	try:
		model = ModelMGPU(model, gpus)
	except:
		pass

	model.compile(optimizer='adam', loss='mean_absolute_error')
	print(model.summary())

	logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)


	early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
	checkpoint = ModelCheckpoint(model_dir+'/best_model.h5', monitor='val_loss', mode='min', \
								verbose=1, save_best_only=True)


	history = model.fit_generator(generator=training_generator,
								validation_data=validation_generator,
								steps_per_epoch=steps_per_epoch,
								epochs=epochs,
								use_multiprocessing=False,
								callbacks=[tensorboard_callback, early_stopping, checkpoint]
								)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-f', '--feature_path', help='Feature path (path containing the saved audio \
						and frames')	
	parser.add_argument('-e', '--epochs', default=20, help='No of epochs to train the model')
	parser.add_argument('-spe', '--steps_per_epoch', default=1000, help='No of steps per epoch')
	parser.add_argument('-g', '--no_of_gpus', default=1, required=False, help='No of GPUs')
	parser.add_argument('-s', '--no_of_still_images', default=1, required=False, \
						help='No of still images')
	parser.add_argument('-md', '--model_directory', default='saved_models', \
						help='Path to save the model')

	args = parser.parse_args()


	print("---------------------")
	feature_path = args.feature_path
	folders = get_folders(feature_path)
	num_folders = len(folders)
	print("Total number of folders = ", num_folders)


	files = []
	for folder in folders:
		sub_folder_path = join(feature_path, folder)
		sub_folders = get_folders(sub_folder_path)
		for sub_folder in sub_folders:
			file_path = join(sub_folder_path, sub_folder)
			file = get_files(file_path, extension=['.jpg'])
			files.extend(file)

		
	model(files, int(args.epochs), int(args.steps_per_epoch), int(args.no_of_gpus), \
			int(args.no_of_still_images), args.model_directory)

	
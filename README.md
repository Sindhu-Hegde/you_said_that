# Speech2Vid

Implementation of [You said that?](https://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17b/chung17b.pdf) in Python.

### Prerequisites
 - Python3
 - cuda
 - ffmpeg
 - librosa
 - OpenCV
 - dlib
 
 ### Overview of the Project
 This is a project on generating a video of a talking face. For any given audio segment and a face image, the method generates a video of the input face lip-synched with the given input audio. There are 2 major components in this method:
 
 - **Data Preprocessing:** The dataset used in this project is [LRS2](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) dataset. Initially, the raw audio data and the frames are extracted from the input video. The Mel Frequency Cepstral Coefficients (MFCC) features are extracted (using ``librosa``) for every 0.35 secs audio data. The ground truth frame (center frame) and the feature frame (random frame far from the ground truth frame) are selected for every 0.35 secs window. The lower half of the ground truth frame is masked and concatenated with feature frame. 
 - **Architecture:** The encoder-decoder CNN model is built to generate the talking video. The audio features and the identity features are used to construct the audio and the identity encoders. the output of the audio and the identity encoders are concatenated and given to the image decoder which generates the output frame.

### Usage

 

 1. **Feature extraction:** Extract and store the audio and identity features
 
	`python3 extract_features.py -d=data/`

 2. **Data preparation:** Prepare the training data 
 
	`python3 prepare_data.py -f=features/ -p=processed_data/`
 
 3. **Train:** Train the model using the processed data
 
  	`python3 train.py -a=audio_features.pkl -i=identity_features.pkl -gt=gt_features.pkl`
  
 4. **Generate video:** Generate the output video for a given audio and image input 
 
 	`python3 predict.py -m=saved_models/model.h5 -t=1.mp4 -f=frame.jpg`

	
Following are the other train parameters that can be tuned: 

    >> python3 train.py
	

    usage: train.py      [-a AUDIO_FEATURE_PATH] [-i IDENTITY_FEATURE_PATH]
                         [-gt GROUND_TRUTH_FEATURE_PATH] [-b BATCH_SIZE]
                         [-se STEPS_PER_EPOCH] [-e NO_OF_EPOCHS] 
                         [-w NO_OF_WORKERS] [-gpu NO_OF_GPUs]
                         
        
    Optional arguments:
        
          -b BATCH_SIZE,		--batch_size                          
				          Batch size of the data to be used while training. Default value is 64.
                                
          -se STEPS_PER_EPOCH,	--steps_per_epoch 
		                          Number of step to be used per epoch while training. Default value is 1000.
		                          
          -e NO_OF_EPOCHS, 		--epochs 
	                                  Number of epochs to be used while training. Default value is 20.
	                                  
          -w NO_OF_WORKERS		--no_of_workers
	                                  Number of workers to be used for multiprocessing execution. Default value is 1.
	                                   
          -gpu NO_OF_GPUs, 		--no_of_gpus
	                                  Total number of GPUs to be used while training. If number of GPUs are greater than 1, then multi-gpu support will be enabled. Default value is 1.
	                                  

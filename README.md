# LipReading
Repo for LipReading research development

## Data
Data is obtained from the LRS3-Ted dataset, produced by T. Afouras, J. S. Chung, A. Zisserman.

From the paper:
"The cropped face tracks are provided as .mp4 files with a
resolution of 224×224 and a frame rate of 25 fps, encoded using
the h264 codec. The audio tracks are provided as single-channel
16-bit 16kHz format..."

"Video preparation. We use a CNN face detector based on the
Single Shot MultiBox Detector (SSD) [10] to detect face appearances in the individual frames.
The time boundaries of a shot are determined by comparing color histograms across consecutive frames [11], and within
each shot, face tracks are generated from face detections based
on their positions.
Audio and text preparation. Only the videos providing english subtitles created by humans were used. The subtitles
in the YouTube videos are broadcast in sync with the audio only at sentence-level, therefore the Penn Phonetics Lab
Forced Aligner (P2FA) [12] is used to obtain a word-level alignment between the subtitle and the audio signal. The alignment
is double-checked against an off-the-shelf Kaldi-based ASR
model."

More information on the dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)
## Code

### pipeline.py
This file contains the preprocssing steps necessary to convert the raw mp4 files into network inputs.
The network uses video frames sampled at 25 Hz and audio signals converted to time vs. freq spectrograms.
These inputs are stored in hdf5 files for use by the network. 

The user must have download the data set linked above and store in the '.data/' directory.

### net.py
This file contains the TensorFlow network class to be used for training and inference. 

### utils.py
Network utilities are stored separately such as input data batch generation.

### run.py
Master training script that calls network training from the command line. Allows the user to use
command line argument parsing for important hyperparameters and training parameters such as 
batch size, learning rate, epochs, etc.

import os
import numpy as np
import subprocess as sp
import json
from scipy import signal
from scipy.io import wavfile
import cv2
import h5py

def gen_vids_dict(main_path, min_length=None):
	# Get directories where clips are found - ignore hidden files
	vid_dirs = [x for x in os.listdir(main_path) if not x.startswith('.')]

	videos = {}
	for vid_dir in vid_dirs:
		clips = [x.split('.')[0] for x in os.listdir(main_path+vid_dir) if x.endswith('.mp4')]
		for clip in clips:
			clip_path = main_path+vid_dir+'/'+clip+'.mp4'

			# Extract vid info if vid is long enough or no min_length specified
			# Avoid probing duration if possible - costly operation
			if min_length is None:
				contents = get_vid_info(main_path, vid_dir, clip)
				videos[(vid_dir, clip)] = contents
			# Check video length to make sure it is long enough
			elif probe_duration(clip_path) >= min_length:
				contents = get_vid_info(main_path, vid_dir, clip)
				videos[(vid_dir, clip)] = contents
			else:
				pass

	# Return a dictionary of video names and corresponding metadata
	return videos

def get_vid_info(main_path, vid_dir, clip):
	meta_path = main_path+vid_dir+'/'+clip+'.txt'

	# Read metadata for video into a dictionary
	with open(meta_path) as fo:
		contents = {}
		for line in fo:
			key, val = line.split(': ')
			val = str.strip(val)
			contents[key] = val
	return contents

def extract_audio(data_path, videos, ac=1, ar=8000):
	'''
	This function extracts .wav files from the raw .mp4 files
	and places them in a separate audio/ directory
	:param data_path: Path to upper level data directory
	:param videos: List of tuples of the form (vid_id, clip_id)
	:param ac: Number of audio channels
	:param ar: Audio frequency of output file
	:return: None 
	'''
	audio_path = data_path + 'audio/'

	# Create directory for processed audio files
	if not os.path.exists(audio_path):
		print('Adding the audio data directory')
		os.system('mkdir {}'.format(audio_path))

	for (vid_id, clip_id) in videos:
		# Generate file names
		fname = vid_id+'_'+clip_id
		raw_file = data_path+'raw/'+vid_id+'/'+clip_id+'.mp4'
		audio_file = audio_path+fname+'.wav'

		# issue command to extract .wav from .mp4
		cmd = 'ffmpeg -i {} -ac {} -ar {} {}'.format(raw_file, ac, ar, audio_file)
		os.system(cmd)

	return None

def probe_duration(vid_file_path):
    '''
	Probe an mp4 file for its length in seconds.
	:param vid_file_path: The absolute (full) path of the video file
	:return duration: Duration of video, in seconds
    '''
    command = ["ffprobe", "-loglevel",  "quiet", "-print_format", "json",
            	"-show_format", "-show_streams", vid_file_path]

    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
    out, err = pipe.communicate()
    json_out = json.loads(out)
    duration = json_out['streams'][0]['duration']

    return float(duration)

def gen_hdf5(hdf5_path, wav_path, vid_path, clip_len, vid_sample_rate,
			 height, width, channels):
	num_frames = clip_len * vid_sample_rate
	frames = gen_frames(clip_path, num_frames, height, width, channels)
	spec = gen_spectrogram(wav_path)

	with h5py.File(hdf5_path, 'w') as f:
		f.create_dataset('frames', data=frames)
		f.create_dataset('spectrogram', data=spec)
	
	return None

def gen_frames(clip_path, num_frames, height, width, channels):

	# Initialize frames array
	frames = np.zeros(shape=[num_frames, height, width, channels])

	# Initialize while loop
	success = True
	count = 0
	cap = cv2.VideoCapture(vid_path)
	while success and count<num_frames:
		success, frame = cap.read()
		
		# Resize if necessary
		if frame.shape != (height, width, channels):
			frame = cv2.resize(frame, (height, width))
		
		# Store frame and move to next
		frames[count] = frame
		count += 1

	if not success:
		print('Error generating frames from {}'.format(clip_path))

	return frames


	

def gen_spectrogram(wav_path):
	'''
	Generate a spectrogram from a .wav file
	:param wav_path: absolute (full) path to the wav file
	:return frequencies: list of frequencies in spectrogram
	:return times: list of times in the spectrogram
	:return spectrogram: spectrogram as ndarray
	'''
	sample_rate, samples = wavfile.read(wav_path)
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

	return frequencies, times, spectrogram
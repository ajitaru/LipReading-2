import os
from pathlib import Path
import numpy as np
import subprocess as sp
import json
from scipy import signal
from scipy.io import wavfile
import cv2
import h5py

def main(num_vids, length=3.0):
	p = Path.cwd()
	data_path = p.joinpath('data')
	raw_data_path = p.joinpath('data', 'raw')

	# Get the locations of all .mp4 files
	clips = gen_vids_dict(data_path=raw_data_path,
						  max_samples=num_vids, min_length=length)
	print('Found {} video clips'.format(len(clips)))

	# Create directory for final processed files
	h5_path = data_path.joinpath('hdf5')
	if not h5_path.exists():
		print('Adding the hdf5 directory')
		h5_path.mkdir()
	
	# Create hdf5 file for each clip
	for i, clip in enumerate(clips.keys()):
		fname = clip.parent.name + '_' + clip.stem

		# Generate .wav file from .mp4 files
		gen_audio_file(data_path, clip, length, 1, 8000)

		audio_file = data_path.joinpath('audio',fname).with_suffix('.wav')
		h5_file = h5_path.joinpath(fname).with_suffix('.hdf5')
		if not h5_file.exists():
			gen_hdf5(h5_file, audio_file, clip, length, 25, 128, 128, 3)
		
		if i+1 >= num_vids:
			break
	return None

def gen_vids_dict(data_path, max_samples, min_length=None):
	'''
	Iterates through a directory of data and extracts paths to each video clip
	as well as metadata about that video clip.
	:param data_path: high level directory where all data is stored
	:param min_length: Video clips shorter than this threshold will not be returned.
					   Designed to ignore very short clips.
	:return videos: Dictionary of video clips and corresponding metadata
	'''
	# Get directories where clips are found - ignore hidden files
	vid_dirs = [x for x in data_path.iterdir() if x.is_dir()]

	videos = {}
	for vid_dir in vid_dirs:
		clip_paths = [x for x in vid_dir.glob('*.mp4')]

		for clip_path in clip_paths:
			# Exit function if max samples have already been encountered
			if len(videos)>=max_samples:
				return videos

			# Extract vid info if vid is long enough or no min_length specified
			# Avoid probing duration if possible - costly operation
			if min_length is None:
				contents = get_vid_info(clip_path)
				videos[clip_path] = contents
			# Check video length to make sure it is long enough
			elif probe_duration(clip_path) >= min_length:
				contents = get_vid_info(clip_path)
				videos[clip_path] = contents
			else:
				pass

	# Return a dictionary of clip_path and corresponding metadata
	return videos

def get_vid_info(clip_path):
	'''
	This function extracts metadata for a video clip by reading
	the txt file associated with a given video clip. 
	:param clip_path: Path to a specific mp4 video clip
	:return contents: Dictionary where key value pairs are metadata
					  for the video clip.
	'''
	meta_path = clip_path.with_suffix('.txt')
	# Read metadata for video into a dictionary
	with open(meta_path) as fo:
		contents = {}
		for line in fo:
			# Extract metadata for video
			key, val = line.split(': ')
			val = str.strip(val)
			contents[key] = val
	return contents

def gen_audio_file(data_path, clip_path, length, ac=1, ar=8000):
	'''
	This function extracts .wav files from the raw .mp4 files
	and places them in a separate audio/ directory
	:param data_path: Path to upper level data directory
	:param videos: List of tuples of the form (vid_id, clip_id)
	:param ac: Number of audio channels
	:param ar: Audio frequency of output file
	:return: None 
	'''
	#audio_path = data_path + 'audio/'
	audio_path = data_path.joinpath('audio')

	# Create directory for processed audio files
	if not audio_path.exists():
		print('Adding the audio data directory')
		audio_path.mkdir()

	# Generate file names
	raw_file = str(clip_path)

	fname = clip_path.parent.name + '_' + clip_path.stem
	audio_file = audio_path.joinpath(fname).with_suffix('.wav')

	if not audio_file.exists():
		# issue command to extract .wav from .mp4
		cmd = 'ffmpeg -loglevel error -i {} -t {} -ac {} -ar {} {}'.\
								format(raw_file, length, ac, ar, audio_file)
		os.system(cmd)

	return None

def probe_duration(vid_file_path):
    '''
	Probe an mp4 file for its length in seconds.
	:param vid_file_path: The absolute (full) path of the video file
	:return duration: Duration of video, in seconds
    '''
    command = ["ffprobe", "-loglevel",  "quiet", "-print_format", "json",
            	"-show_format", "-show_streams", str(vid_file_path)]

    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
    out, _ = pipe.communicate()
    json_out = json.loads(out)
    duration = json_out['streams'][0]['duration']

    return float(duration)

def gen_hdf5(hdf5_path, wav_path, clip_path, clip_len, vid_sample_rate,
			 height, width, channels):
	num_frames = int(clip_len * vid_sample_rate)
	frames = gen_frames(clip_path, num_frames, height, width, channels)
	_, _, spec = gen_spectrogram(wav_path, clip_len)

	with h5py.File(str(hdf5_path), 'w') as f:
		f.create_dataset('frames', data=frames)
		f.create_dataset('spectrogram', data=spec)
	
	return None

def gen_frames(clip_path, num_frames, height, width, channels):
	'''
	Extract image frame array from an .mp4 file
	:param clip_path: Path to the .mp4 file
	:param num_frames: The desired number of frames to extract from the video
	:param height: Height of the image frame
	:param width: Width of the image frame
	:param channels: number of channels in the image encoding (typically 3)
	:return frames: NumPy array of shame (num_frames, height, width, num_channels)
	'''
	# Initialize frames array
	frames = np.zeros(shape=[num_frames, height, width, channels], dtype=np.float32)

	# Initialize while loop
	success = True
	count = 0
	cap = cv2.VideoCapture(str(clip_path))
	while success and count<num_frames:
		success, frame = cap.read()
		if success:
			# Resize if necessary
			if frame.shape != (height, width, channels):
				frame = cv2.resize(frame, (height, width))
			
			# Store frame and move to next
			frames[count] = frame
			count += 1

	if not success:
		print('Error generating frames from {}'.format(str(clip_path)))

	return frames

def gen_spectrogram(wav_path, length):
	'''
	Generate a spectrogram from a .wav file
	:param wav_path: absolute (full) path to the wav file
	:return frequencies: list of frequencies in spectrogram
	:return times: list of times in the spectrogram
	:return spectrogram: spectrogram as ndarray
	'''
	sample_rate, samples = wavfile.read(str(wav_path))

	# Cap number of samples for desired clip length
	num_samples = int(sample_rate * length)
	samples = samples[:num_samples]

	# Generate spectrogram
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
	return frequencies, times, spectrogram

if __name__ == '__main__':
	main(500, 3.0)
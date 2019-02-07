import tensorflow as tf
import numpy as np
from pathlib import Path
import random
import h5py

def batch_gen(batch_size, data_path):
	files = [x for x in data_path.glob('*.hdf5')]
	files = files[:10]
	with h5py.File(files[0], 'r') as f:
		# Extract the shape of the 'frames' and 'spectrogram' tensors
		frames_shape = f['frames'].shape
		frames_shape = (batch_size, *frames_shape)
		spec_shape = f['spectrogram'].shape
		spec_shape = (batch_size, *spec_shape)

	# Initialize numpy arrays
	vid_in = np.zeros(shape=frames_shape, dtype=np.float32)
	aud_in = np.zeros(shape=spec_shape, dtype=np.float32)

	num_batches = int(np.floor(len(files)/batch_size))
	while True:
		for n in range(num_batches):
			batch_files = files[n*batch_size:(n+1)*batch_size]
			for idx, fpath in enumerate(batch_files):
				with h5py.File(fpath, 'r') as f:
					vid_in[idx] = f['frames']
					aud_in[idx] = f['spectrogram']
			yield vid_in, aud_in
		random.shuffle(files)

class net():
	def __init__(self):
		# Define inputs

		# TODO - confirm spectrogram input dimensions
		self.vid_in = tf.placeholder(dtype=tf.float32, shape=[None, 75, 128, 128, 3])
		self.aud_in = tf.placeholder(dtype=tf.float32, shape=[None, 129, 107, 3])
		self.is_training = tf.placeholder(dtype=tf.bool)

	def create_network(self):
		self.audio_branch()
		self.video_branch()	

		# TODO - Confirm axis of cosine similarity.
		# Consider implementing function fro scratch
		self.loss = tf.losses.cosine_distance(self.aud_out, self.vid_out, axis=1)

	def audio_branch(self):
		# Convolutional layer with 32 filters and a kernel size of 5
		aud_conv_1 = tf.layers.conv2d(self.aud_in, 32, 5, 
						activation=None, kernel_initializer='he_normal')
		aud_conv_1 = tf.layers.batch_normalization(aud_conv_1, training=self.is_training)
		aud_conv_1 = tf.nn.leaky_relu(aud_conv_1, alpha=0.2)
		# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
		aud_conv_1 = tf.layers.max_pooling2d(aud_conv_1, 2, 2)

		# Convolutional layer with 32 filters and a kernel size of 5
		aud_conv_2 = tf.layers.conv2d(aud_conv_1, 32, 5, 
						activation=None, kernel_initializer='he_normal')
		aud_conv_2 = tf.layers.batch_normalization(aud_conv_2, training=self.is_training)
		aud_conv_2 = tf.nn.leaky_relu(aud_conv_2, alpha=0.2)
		# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
		aud_conv_2 = tf.layers.max_pooling2d(aud_conv_2, 2, 2)

		aud_flat = tf.layers.flatten(aud_conv_2)
		
		self.aud_out = tf.layers.dense(aud_flat, 100, activation=None, use_bias=True)

	def video_branch(self):
		# Convolutional layer with 32 filters and a kernel size of 3
		vid_conv_1 = tf.layers.conv3d(self.vid_in, 32, 3,
						activation=None, kernel_initializer='he_normal')
		vid_conv_1 = tf.layers.batch_normalization(vid_conv_1, training=self.is_training)
		vid_conv_1 = tf.nn.leaky_relu(vid_conv_1, alpha=0.2)
		vid_conv_1 = tf.layers.max_pooling3d(vid_conv_1, pool_size=(2,2,1), strides=1)

		# Convolutional layer with 32 filters and a kernel size of 3
		vid_conv_2 = tf.layers.conv3d(vid_conv_1, 32, 3,
						activation=None, kernel_initializer='he_normal')
		vid_conv_2 = tf.layers.batch_normalization(vid_conv_2, training=self.is_training)
		vid_conv_2 = tf.nn.leaky_relu(vid_conv_2, alpha=0.2)
		vid_conv_2 = tf.layers.max_pooling3d(vid_conv_2, pool_size=(2,2,1), strides=1)

		vid_flat = tf.layers.flatten(vid_conv_2)

		self.vid_out = tf.layers.dense(vid_flat, 100, activation=None, use_bias=True)
	
	def train(self, batch_size, lr, iters, display_step):
		self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		train_op = self.optimizer.minimize(self.loss)

		hdf5_path = Path.cwd().joinpath('data', 'hdf5')
		gen = batch_gen(batch_size, hdf5_path)

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for i in range(iters):
				batch_vid, batch_aud = next(gen)
				
				loss = sess.run(train_op, feed_dict={self.vid_in: batch_vid,
											  self.aud_in: batch_aud,
											  self.is_training: True})

				if i%display_step==0:
					print('The loss at iteration {} is {}'.format(i, loss))
				'''
				if i%display_step==0:
					# Calculate batch loss and accuracy
					loss = sess.run([self.loss], feed_dict={self.vid_in: vid_test,
															self.aud_in: aud_test,
															self.is_training: False})
				'''

if __name__ == '__main__':
	model = net()
	model.create_network()

	model.train(2, 0.001, 10, 1)
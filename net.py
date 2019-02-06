import tensorflow as tf
import numpy as np


class net():
	def __init__(self):
		# Define inputs

		# TODO - confirm spectrogram input dimensions
		self.spec_in = tf.placeholder(dtype=tf.float32, shape=[None, 300, 128, 3])
		self.vid_in = tf.placeholder(dtype=tf.float32, shape=[None, 75, 128, 128, 3])
		self.training = tf.placeholder(dtype=tf.bool)

	def create_network(self):
		self.audio_branch()
		self.video_branch()	

		# TODO - Confirm axis of cosine similarity.
		# Consider implementing function fro scratch
		self.loss = tf.losses.cosine_distance(self.aud_out, self.vid_out, axis=1)

	def audio_branch(self):
		# Convolutional layer with 32 filters and a kernel size of 5
		aud_conv_1 = tf.layers.conv2d(self.spec_in, 32, 5, 
						activation=None, kernel_initializer='he_normal')
		aud_conv_1 = tf.layers.batch_normalization(aud_conv_1, training=self.training)
		aud_conv_1 = tf.nn.leaky_relu(aud_conv_1, alpha=0.2)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
		aud_conv_1 = tf.layers.max_pooling2d(aud_conv_1, 2, 2)

		# Convolutional layer with 32 filters and a kernel size of 5
		aud_conv_2 = tf.layers.conv2d(aud_conv_1, 32, 5, 
						activation=None, kernel_initializer='he_normal')
		aud_conv_2 = tf.layers.batch_normalization(aud_conv_2, training=self.training)
		aud_conv_2 = tf.nn.leaky_relu(aud_conv_2, alpha=0.2)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
		aud_conv_2 = tf.layers.max_pooling2d(aud_conv_2, 2, 2)

		aud_flat = tf.layers.flatten(aud_conv_2)
		
		self.aud_out = tf.layers.dense(aud_flat, 100, activation=None, use_bias=True)

	def video_branch(self):
		# Convolutional layer with 32 filters and a kernel size of 3
		vid_conv_1 = tf.layers.conv3d(self.vid_in, 32, 3,
						activation=None, kernel_initializer='he_normal')
		vid_conv_1 = tf.layers.batch_normalization(vid_conv_1, training=self.training)
		vid_conv_1 = tf.nn.leaky_relu(vid_conv_1, alpha=0.2)
		vid_conv_1 = tf.layers.max_pooling3d(vid_conv_1, pool_size=(2,2,1), strides=1)

		# Convolutional layer with 32 filters and a kernel size of 3
		vid_conv_2 = tf.layers.conv3d(vid_conv_1, 32, 3,
						activation=None, kernel_initializer='he_normal')
		vid_conv_2 = tf.layers.batch_normalization(vid_conv_2, training=self.training)
		vid_conv_2 = tf.nn.leaky_relu(vid_conv_2, alpha=0.2)
		vid_conv_2 = tf.layers.max_pooling3d(vid_conv_2, pool_size=(2,2,1), strides=1)

		vid_flat = tf.layers.flatten(vid_conv_2)

		self.vid_out = tf.layers.dense(vid_flat, 100, activation=None, use_bias=True)
	
	def train():
		#self.optimizer = 
		pass

if __name__ == '__main__':
	model = net()
	model.create_network()

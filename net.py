import tensorflow as tf
import numpy as np
from utils import batch_gen, print_tensor_shape

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
		# Consider implementing function from scratch
		self.loss = tf.losses.cosine_distance(self.aud_out, self.vid_out, axis=1)

	def audio_branch(self):
		# TODO Add tensor shape outputs
		with tf.name_scope('aud_conv_1'):
			# Convolutional layer with 32 filters and a kernel size of 5
			aud_conv_1 = tf.layers.conv2d(self.aud_in, 32, (5,5), padding='same', 
							activation=None, kernel_initializer='he_normal')
			aud_conv_1 = tf.layers.batch_normalization(aud_conv_1, training=self.is_training)
			aud_conv_1 = tf.nn.leaky_relu(aud_conv_1, alpha=0.2)
			# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
			aud_conv_1 = tf.layers.max_pooling2d(aud_conv_1, 2, 2)
		print_tensor_shape(aud_conv_1, 'Output shape of audio convolution 1: ')

		with tf.name_scope('aud_conv_2'):
			# Convolutional layer with 32 filters and a kernel size of 5
			aud_conv_2 = tf.layers.conv2d(aud_conv_1, 32, (5,5), padding='same',
							activation=None, kernel_initializer='he_normal')
			aud_conv_2 = tf.layers.batch_normalization(aud_conv_2, training=self.is_training)
			aud_conv_2 = tf.nn.leaky_relu(aud_conv_2, alpha=0.2)
			# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
			aud_conv_2 = tf.layers.max_pooling2d(aud_conv_2, 2, 2)
		print_tensor_shape(aud_conv_2, 'Output shape of audio convolution 2: ')

		aud_flat = tf.layers.flatten(aud_conv_2)
		print_tensor_shape(aud_flat, 'Output shape of aud flattened layer: ')
		
		self.aud_out = tf.layers.dense(aud_flat, 100, activation=None, use_bias=True)
		print_tensor_shape(self.aud_out, 'Output shape of audio branch: ')

	def video_branch(self):
		# TODO Add tensor shape outputs
		with tf.name_scope('vid_conv_1'):
			# Convolutional layer with 32 filters and a kernel size of 3
			vid_conv_1 = tf.layers.conv3d(self.vid_in, 32, (3,3,3), padding='same',
							activation=None, kernel_initializer='he_normal')
			vid_conv_1 = tf.layers.batch_normalization(vid_conv_1, training=self.is_training)
			vid_conv_1 = tf.nn.leaky_relu(vid_conv_1, alpha=0.2)
			vid_conv_1 = tf.layers.max_pooling3d(vid_conv_1, pool_size=(2,2,1), strides=1)
		print_tensor_shape(vid_conv_1, 'Output shape of vid convolution 1: ')

		with tf.name_scope('vid_conv_2'):
			# Convolutional layer with 32 filters and a kernel size of 3
			vid_conv_2 = tf.layers.conv3d(vid_conv_1, 32, (3,3,3), padding='same',
							activation=None, kernel_initializer='he_normal')
			vid_conv_2 = tf.layers.batch_normalization(vid_conv_2, training=self.is_training)
			vid_conv_2 = tf.nn.leaky_relu(vid_conv_2, alpha=0.2)
			vid_conv_2 = tf.layers.max_pooling3d(vid_conv_2, pool_size=(2,2,1), strides=1)
		print_tensor_shape(vid_conv_2, 'Output shape of vid convolution 2: ')

		vid_flat = tf.layers.flatten(vid_conv_2)
		print_tensor_shape(vid_flat, 'Output shape of vid flattened layer: ')

		self.vid_out = tf.layers.dense(vid_flat, 100, activation=None, use_bias=True)
		print_tensor_shape(self.vid_out, 'Output shape of video branch: ')
	
	def train(self, batch_size, lr, epochs):
		self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		train_op = self.optimizer.minimize(self.loss)

		gen = batch_gen(batch_size)

		# Determine number of iterations
		data_path = Path.cwd().joinpath('data', 'hdf5')
		files = [x for x in data_path.glob('*.hdf5')]
		iters = int(np.floor(epochs*len(files)/batch_size))

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)

			print('Training...')
			e=1
			for i in range(iters):
				batch_vid, batch_aud = next(gen)
				sess.run(self.train_op, feed_dict={self.vid_in: batch_vid,
											  self.aud_in: batch_aud,
											  self.is_training: True})
				if i%(len(files)/batch_size)==0:
					print('Epoch {} finished'.format(e))
					e += 1

				# TODO add validation testing at certain intervals.
				# TODO add early stopping and saving of model checkpoints
				# TODO Improve output to user

	def inference():
		'''TODO Create a function that will take  in a set of frames
		and spectrogram and determine whether or not they are correlated.
		The inference shall output 1 if they are a match, and 0 if they
		are not a match. The function shall do this operation for a specified
		number of examples and return the predictions in the form of an array
		'''
		pass

if __name__ == '__main__':
	model = net()
	model.create_network()
	#model.train(1, 0.001, 1, 1)
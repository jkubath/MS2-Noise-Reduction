# noise_reduction_ms2.py
# source: https://www.easy-tensorflow.com/tf-tutorials/autoencoders/noise-removal
# command: python noise_reduction_ms2.py
# note: change logs_path on line 22 to your output directory

# Goal: change the implementation from the MNIST data to read in noisy ms2
#   data and output denoised ms2 data.  This will allow database search
#   algorithms to have fewer peaks to check against.

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import protein # helper for reading ms2 data and binning to data[spectra][peak data]
import os # used to access operating system directory structure
import sys

def plot_images(original_images, noisy_images, reconstructed_images):
	"""
	Create figure of original and reconstructed image.
	:param original_image: original images to be plotted, (?, img_h*img_w)
	:param noisy_image: original images to be plotted, (?, img_h*img_w)
	:param reconstructed_image: reconstructed images to be plotted, (?, img_h*img_w)
	"""
	num_images = 3
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=num_images, ncols=1)
	fig.subplots_adjust(hspace=.1, wspace=0)

	img_h = 1
	img_w = 6000
	# Plot image.
	ax1.plot(original_images[0])
	ax2.plot(noisy_images[0])
	ax3.plot(reconstructed_images[0])

	ax1.set_title("Original Image")
	ax2.set_title("Noisy Image")
	ax3.set_title("Reconstructed Image")

	fig.tight_layout()
	plt.show()


print("Noise Reduction NN")
# Read in the data
filePath = str(os.getcwd()) + "/" + sys.argv[1] + "/"

# length of peak array (0 to max m/z value)
dataLength = 6000

# training set
#-------------------------------------------------------------------------
train_no_noise_file = "train_no_noise.ms2" # ms2 file containing with no noise
train_noise_file = "train_noise.ms2" # regular ms2 data

train_no_noise_output_file = "train_no_noise_binned.ms2"
train_noise_output_file = "train_noise_binned.ms2"

train_write_noise_output = False
train_write_no_noise_output = False

# validation set
#-------------------------------------------------------------------------
valid_no_noise_file = "valid_no_noise.ms2" # ms2 file containing with no noise
valid_noise_file = "valid_noise.ms2" # regular ms2 data

valid_no_noise_output_file = "valid_no_noise_binned.ms2"
valid_noise_output_file = "valid_noise_binned.ms2"

valid_write_noise_output = False
valid_write_no_noise_output = False

# Create file readers for training set
#-------------------------------------------------------------------------
print("Creating training file objects")
try:
	# labels
	train_noise_file_object = open(filePath + train_noise_file, "r")
	# input data
	train_no_noise_file_object = open(filePath + train_no_noise_file, "r")

	# write to the output files
	train_no_noise_output_file = open(filePath + train_no_noise_output_file, "w")
	train_noise_output_file = open(filePath + train_noise_output_file, "w")
except (OSError, IOError) as e:
	print(e)
	exit()

# Create file readers for validation set
#-------------------------------------------------------------------------
print("Creating validation file objects")
try:
	# labels
	valid_noise_file_object = open(filePath + valid_noise_file, "r")
	# input data
	valid_no_noise_file_object = open(filePath + valid_no_noise_file, "r")

	# write to the output files
	valid_noise_output_file = open(filePath + valid_noise_output_file, "w")
	valid_no_noise_output_file = open(filePath + valid_no_noise_output_file, "w")
except (OSError, IOError) as e:
	print(e)
	exit()

# Read and output training data
# returns numpy array of spectrum data binned to nearest m/z integer value
# writes the data in csv format to file
#-------------------------------------------------------------------------
print("Reading training data")
train_noise_data = protein.readFile(train_noise_file_object, dataLength, train_noise_output_file, train_write_noise_output)
train_no_noise_data = protein.readFile(train_no_noise_file_object, dataLength, train_no_noise_output_file, train_write_no_noise_output)

print("Training data shape")
print("noise: {}".format(train_noise_data.shape))
print("no_noise: {}".format(train_noise_data.shape))

# Read and output validation data
# returns numpy array of spectrum data binned to nearest m/z integer value
# writes the data in csv format to file
#-------------------------------------------------------------------------
print("Reading validation data")
valid_noise_data = protein.readFile(valid_noise_file_object, dataLength, valid_noise_output_file, valid_write_noise_output)
valid_no_noise_data = protein.readFile(valid_no_noise_file_object, dataLength, valid_no_noise_output_file, valid_write_no_noise_output)

print("Validation data shape")
print("noise: {}".format(valid_noise_data.shape))
print("no_noise: {}".format(valid_no_noise_data.shape))

# Print first 10 objects in data
#-------------------------------------------------------------------------
# i = 0
# for x in np.nditer(noise_data):
#   print(x),
#   i += 1
#   if i == 10:
#       break

# i = 0
# for x in np.nditer(no_noise_data):
#   print(x),
#   i += 1
#   if i == 10:
#       break

# i = 0
# for x in np.nditer(valid_noise_data):
#   print(x),
#   i += 1
#   if i == 10:
#       break

# i = 0
# for x in np.nditer(valid_no_noise_data):
#   print(x),
#   i += 1
#   if i == 10:
#       break

# https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
# hyper-parameters
logs_path = str(os.getcwd()) # path to the folder that we want to save the logs for Tensorboard
learning_rate = 0.001  # The optimization learning rate
epochs = 600 #10  # Total number of training epochs
batch_size = 600 #100  # Training batch size
display_freq = 100 #100  # Frequency of displaying the training results
# number of units in the hidden layer
h1 = 200

# Create TensorFlow Dataset object for training data
#-------------------------------------------------------------------------
# First input argument is Tensor objects of the noise data (training)
# Second input argument is Tensor objects of the no noise data (labels)
tf_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(train_noise_data, dtype=tf.float32),
	tf.convert_to_tensor(train_no_noise_data, dtype=tf.float32))).shuffle(100).repeat().batch(batch_size)

# Create TensorFlow Dataset object for validation data
#-------------------------------------------------------------------------
# First input argument is Tensor objects of the noise data (training)
# Second input argument is Tensor objects of the no noise data (labels)
valid_tf_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(valid_noise_data,dtype=tf.float32), 
	tf.convert_to_tensor(valid_no_noise_data, dtype=tf.float32))).shuffle(100).repeat().batch(batch_size)

# Create general iterator
iterator = tf.data.Iterator.from_structure(tf_data.output_types, tf_data.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately,
# but using the same structure via the common iterator
training_init_op = iterator.make_initializer(tf_data)
validation_init_op = iterator.make_initializer(valid_tf_data)

# tf_iter = tf_data.make_one_shot_iterator()
# x, y = tf_iter.get_next()

# print("TF_data {}".format(x.shape))

def nn_model(in_data):
	# bn = tf.layers.batch_normalization(in_data)
	fc1 = tf.layers.dense(in_data, h1)
	fc2 = tf.layers.dense(fc1, h1)
	fc3 = tf.layers.dense(fc2, h1)
	fc4 = tf.layers.dense(fc3, h1)
	fc5 = tf.layers.dense(fc4, h1)
	fc6 = tf.layers.dense(fc5, h1)
	fc7 = tf.layers.dense(fc6, h1)
	fc8 = tf.layers.dense(fc7, h1)
	fc9 = tf.layers.dense(fc8, h1)
	fc10 = tf.layers.dense(fc9, h1)
	fc11 = tf.layers.dense(fc10, h1)
	drop = tf.layers.dropout(fc10)
	fc12 = tf.layers.dense(drop, dataLength)
	return fc12

# create the neural network model
logits = nn_model(next_element[0])

# add the optimizer and loss
# loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
# loss = tf.reduce_sum(tf.losses.mean_squared_error(next_element[1], logits))
loss = tf.reduce_sum(tf.losses.absolute_difference(next_element[1], logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# get accuracy
# prediction = tf.argmax(logits, 1)
# equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
# accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
accuracy = tf.losses.mean_squared_error(next_element[1], logits)

init_op = tf.global_variables_initializer()

# run the training
with tf.Session() as sess:
	sess.run(init_op)
	sess.run(training_init_op)
	for i in range(epochs):
		l, _, acc = sess.run([loss, optimizer, accuracy])
		if i % 50 == 0:
			print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
	# now setup the validation run
	valid_iters = 100
	# re-initialize the iterator, but this time with validation data
	sess.run(validation_init_op)
	avg_acc = 0
	for i in range(valid_iters):
		acc = sess.run([loss])
		avg_acc += acc[0]
	print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters, (avg_acc / valid_iters) * 100))

	# Test the network after training

	# test set
	#-------------------------------------------------------------------------
	test_no_noise_file = "test_no_noise.ms2" # ms2 file containing with no noise
	test_noise_file = "test_noise.ms2" # regular ms2 data

	test_no_noise_output_file = "test_no_noise_binned.ms2"
	test_noise_output_file = "test_noise_binned.ms2"

	test_write_noise_output = False
	test_write_no_noise_output = False

	# length of peak array
	dataLength = 6000


	# Create file readers for test set
	#-------------------------------------------------------------------------
	print("Creating test file objects")
	try:
		# input data
		test_no_noise_file_object = open(filePath + test_no_noise_file, "r")
		# labels
		test_noise_file_object = open(filePath + test_noise_file, "r")

		# write to the output files
		test_no_noise_output_file = open(filePath + test_no_noise_output_file, "w")
		test_noise_output_file = open(filePath + test_noise_output_file, "w")
	except (OSError, IOError) as e:
		print(e)
		exit()

	# Read and output test data
	#-------------------------------------------------------------------------
	print("Reading test data")
	test_no_noise_data = protein.readFile(test_no_noise_file_object, dataLength, test_noise_output_file, test_write_noise_output)
	test_noise_data = protein.readFile(test_noise_file_object, dataLength, test_noise_output_file, test_write_noise_output)

	print(len(test_noise_data))
	print(len(test_no_noise_data))

	# Create test Tensorflow Dataset
	test_tf_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(test_noise_data,
		dtype=tf.float32), tf.convert_to_tensor(test_no_noise_data, dtype=tf.float32))).repeat().batch(batch_size)

	# Test iterator
	test_init_op = iterator.make_initializer(test_tf_data)

	# now setup the validation run
	test_iters = 100
	# re-initialize the iterator, but this time with validation data
	sess.run(test_init_op)
	avg_acc = 0
	for i in range(test_iters):
		acc = sess.run([accuracy])
		avg_acc += acc[0]
	print('---------------------------------------------------------')
	print("Average test set accuracy over {} iterations is {:.2f}%".format(test_iters, (avg_acc / test_iters) * 100))
	print('---------------------------------------------------------')

	x_reconstruct = sess.run(logits)

	# print(len(x_reconstruct))
	print(x_reconstruct.shape)
	x,y = sess.run(next_element)
	print(x)
	print(y)
	# print(next_element[0].shape)
	# print(next_element[1].shape)

	plot_images(y,x, x_reconstruct)
	
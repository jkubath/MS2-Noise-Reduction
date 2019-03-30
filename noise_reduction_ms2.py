# noise_reduction_ms2.py
# source: https://www.easy-tensorflow.com/tf-tutorials/autoencoders/noise-removal
# command: python noise_reduction_ms2.py
# note: change logs_path on line 22 to your output directory

# Goal: change the implementation from the MNIST data to read in noisy ms2
# 	data and output denoised ms2 data.  This will allow database search
#	algorithms to have fewer peaks to check against.

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import protein # helper for reading ms2 data and binning to data[spectra][peak data]
import os # used to access operating system directory structure
import sys

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

train_write_noise_output = True
train_write_no_noise_output = True

# validation set
#-------------------------------------------------------------------------
valid_no_noise_file = "valid_no_noise.ms2" # ms2 file containing with no noise
valid_noise_file = "valid_noise.ms2" # regular ms2 data

valid_no_noise_output_file = "valid_no_noise_binned.ms2"
valid_noise_output_file = "valid_noise_binned.ms2"

valid_write_noise_output = True
valid_write_no_noise_output = True

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
# 	print(x),
# 	i += 1
# 	if i == 10:
# 		break

# i = 0
# for x in np.nditer(no_noise_data):
# 	print(x),
# 	i += 1
# 	if i == 10:
# 		break

# i = 0
# for x in np.nditer(valid_noise_data):
# 	print(x),
# 	i += 1
# 	if i == 10:
# 		break

# i = 0
# for x in np.nditer(valid_no_noise_data):
# 	print(x),
# 	i += 1
# 	if i == 10:
# 		break

# hyper-parameters
logs_path = str(os.getcwd()) # path to the folder that we want to save the logs for Tensorboard
learning_rate = 0.001  # The optimization learning rate
epochs = 50 #10  # Total number of training epochs
batch_size = 32 #100  # Training batch size
display_freq = 100 #100  # Frequency of displaying the training results

# number of units in the hidden layer
h1 = 50
h2 = 50

# weight and bias wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)

def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(name, shape=[in_dim, num_units])
        tf.summary.histogram('W', W)
        b = bias_variable(name, [num_units])
        tf.summary.histogram('b', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

# Create graph
# Placeholders for inputs (x), outputs(y)
with tf.variable_scope('Input'):
    x_original = tf.placeholder(tf.float32, shape=[None, dataLength], name='X_original')
    x_noisy = tf.placeholder(tf.float32, shape=[None, dataLength], name='X_noisy')

fc1 = fc_layer(x_noisy, h1, 'Hidden_layer_1', use_relu=True)
fc2 = fc_layer(fc1, h2, 'Hiden_layer_2', use_relu=True)
# fc3 = fc_layer(fc2, h3, 'Hiden_layer_3', use_relu=True)
# fc4 = fc_layer(fc3, h4, 'Hiden_layer_4', use_relu=True)
out = fc_layer(fc2, dataLength, 'Output_layer', use_relu=False)

# Define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(x_original, out), name='loss')
        tf.summary.scalar('loss', loss)
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph (session)
sess = tf.InteractiveSession() # using InteractiveSession instead of Session to test network in separate cell
sess.run(init)
train_writer = tf.summary.FileWriter(logs_path, sess.graph)
num_tr_iter = int(len(train_noise_data) / batch_size)
global_step = 0

start = 0
end = 0
index_in_epoch = 0

def next_batch(start, end, index_in_epoch, train_noise_data, train_no_noise_data, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > len(train_noise_data):
        # Finished epoch
        # Shuffle the data
        perm = np.arange(len(train_noise_data))
        np.random.shuffle(perm)
        train_noise_data = train_noise_data[perm]
        train_no_noise_data = train_no_noise_data[perm]

        # Start next epoch
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return start, end, index_in_epoch, train_noise_data, train_no_noise_data, train_noise_data[start:end], train_no_noise_data[start:end]

# Create TensorFlow Dataset object for validation data
#-------------------------------------------------------------------------
# First input argument is Tensor objects of the noise data (training)
# Second input argument is Tensor objects of the no noise data (labels)
#valid_tf_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(valid_noise_data, dtype=tf.float32), tf.convert_to_tensor(valid_no_noise_data, dtype=tf.float32))).repeat().batch(batch_size, True)

#valid_tf_iter = valid_tf_data.make_one_shot_iterator()
#valid_next = valid_tf_iter.get_next()

for epoch in range(epochs):
    # Create TensorFlow Dataset object for training data
    #-------------------------------------------------------------------------
    # First input argument is Tensor objects of the noise data (training)
    # Second input argument is Tensor objects of the no noise data (labels)
    #train_tf_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(train_noise_data,  dtype=tf.float32), tf.convert_to_tensor(train_no_noise_data, dtype=tf.float32))).repeat().batch(batch_size, True)

    #train_tf_iter = train_tf_data.make_one_shot_iterator()
    #train_next = train_tf_iter.get_next()

    print('Training epoch: {}'.format(epoch + 1))
    for iteration in range(num_tr_iter):
        # get the next batch of data
        start, end, index_in_epoch, train_noise_data, train_no_noise_data, train_noise, train_no_noise = next_batch(start, end, index_in_epoch, train_noise_data, train_no_noise_data, batch_size)

        global_step += 1
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x_original: train_no_noise, x_noisy: train_noise})

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch = sess.run(loss, feed_dict={x_original: train_no_noise, x_noisy: train_noise})
            print("iter {0:3d}:\t Reconstruction loss={1:.3f}".
                  format(iteration, loss_batch))

    # Run validation after every epoch
    loss_valid = sess.run(loss, feed_dict={x_original: valid_no_noise_data, x_noisy: valid_noise_data})
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.3f}".
          format(epoch + 1, loss_valid))
    print('---------------------------------------------------------')



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

# Test the network after training
# Make a noisy image

# validation set
#-------------------------------------------------------------------------
test_no_noise_file = "test_no_noise.ms2" # ms2 file containing with no noise
test_noise_file = "test_noise.ms2" # regular ms2 data

test_no_noise_output_file = "test_no_noise_binned.ms2"
test_noise_output_file = "test_noise_binned.ms2"

test_write_noise_output = True
test_write_no_noise_output = True

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

# Reconstruct a clean image from noisy image
x_reconstruct = sess.run(out, feed_dict={x_noisy: test_noise_data})
# Calculate the loss between reconstructed image and original image
loss_test = sess.run(loss, feed_dict={x_original: test_no_noise_data, x_noisy: test_noise_data})
print('---------------------------------------------------------')
print("Test loss of original image compared to reconstructed image : {0:.3f}".format(loss_test))
print('---------------------------------------------------------')

# Plot original image, noisy image and reconstructed image
plot_images(test_no_noise_data, test_noise_data, x_reconstruct)

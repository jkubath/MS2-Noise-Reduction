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
from keras.utils import to_categorical
from keras import models
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten

# Print the data array
def printArray(data, length = 10):
	i = 0
	for x in np.nditer(data):
		print(x),
		i += 1
		if i == length:
			break

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
    img_w = 5000
    # Plot image.
    ax1.plot(original_images[0])
    ax2.plot(noisy_images[0])
    ax3.plot(reconstructed_images[0])

    ax1.set_title("Original Image")
    ax2.set_title("Noisy Image")
    ax3.set_title("Reconstructed Image")

    fig.tight_layout()
    plt.show()

def check_peaks(original_images, noisy_images, reconstructed_images):
	correct_peaks = 0
	missed_peaks = 0
	wrong_peaks = 0

	total_noise = 0
	total_no_noise = 0

	threshold = 0.9

	orig = original_images[0]
	noise = noisy_images[0]
	rec = reconstructed_images[0]

	for index in range(len(orig)):
		# no noise has a peak
		if orig[index] > 0:
			if rec[index] > threshold:
				correct_peaks += 1
			else:
				missed_peaks += 1

			total_no_noise += 1
		# wrong peak
		elif rec[index] > threshold:
			wrong_peaks += 1

		if noise[index] > 0:
			total_noise += 1

	print("Total noise peaks: {}".format(total_noise))
	print("Total no noise peaks: {}".format(total_no_noise))
	print("Correct peaks: {}".format(correct_peaks))
	print("Missed peaks: {}".format(missed_peaks))
	print("Wrong peaks: {}".format(wrong_peaks))

def reconstruct(original_images, noisy_images, reconstructed_images):
	threshold = 0.8
	recon = []

	value = 0
	for index in range(len(reconstructed_images[0])):
		value = reconstructed_images[0][index]
		if value > threshold:
			recon.append(noisy_images[0][index])
		else:
			recon.append(0)

	plot_images(original_images, noisy_images, [recon])

print("Noise Reduction NN")
# Read in the data
filePath = str(os.getcwd()) + "/big_data/"

# length of peak array (0 to max m/z value)
dataLength = 5000

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
# printArray(train_noise_data)
# printArray(train_no_noise_data)
# printArray(valid_noise_data)
# printArray(valid_no_noise_data)

# hyper-parameters
logs_path = str(os.getcwd()) # path to the folder that we want to save the logs for Tensorboard
learning_rate = 0.001  # The optimization learning rate
epochs = 5 #10  # Total number of training epochs
batch_size = 10 #100  # Training batch size
display_freq = 1 #100  # Frequency of displaying the training results

# number of units in the hidden layer
h1 = 100

model = models.Sequential()
# Input - Layer
model.add(Dense(5000, activation = "relu",input_shape=(dataLength,)))
# model.add(Dropout(0.8, noise_shape=None, seed=None))
# model.add(Dense(1000, activation = "relu"))
# model.add(Dropout(0.8, noise_shape=None, seed=None))
# model.add(Dense(50, activation = "relu"))
# Output- Layer
model.add(Dense(dataLength, activation = "sigmoid"))
# model.add(Activation("sigmoid"))
model.summary()
# compiling the model
model.compile(
 optimizer = "adam",
 loss = "mean_squared_error",
 metrics = ["accuracy"]
)
results = model.fit(
 train_noise_data, train_no_noise_data,
 epochs= 20,
 batch_size = 500,
 validation_data = (valid_noise_data, valid_no_noise_data),
 verbose = 2
)

# validation set
# -------------------------------------------------------------------------
test_no_noise_file = "test_no_noise.ms2" # ms2 file containing with no noise
test_noise_file = "test_noise.ms2" # regular ms2 data

test_no_noise_output_file = "test_no_noise_binned.ms2"
test_noise_output_file = "test_noise_binned.ms2"

test_write_noise_output = True
test_write_no_noise_output = True

# length of peak array
dataLength = 5000


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

# make prediction on test data
test_prediction = model.predict(
                x = test_noise_data,
                batch_size=None,
                verbose=1,
                steps=None)

# Calculate the loss between reconstructed image and original image
loss_test = model.evaluate(
                        x = test_prediction,
                        y = test_no_noise_data,
                        verbose = 1,
                    )
print('---------------------------------------------------------')
print(model.metrics_names)
print(loss_test)
# print("Test loss of original image compared to reconstructed image : {0:.3f}".format(loss_test))
print('---------------------------------------------------------')

# Plot original image, noisy image and reconstructed image
# plot_images(test_no_noise_data, test_noise_data, test_prediction)
check_peaks(test_no_noise_data, test_noise_data, test_prediction)
reconstruct(test_no_noise_data, test_noise_data, test_prediction)

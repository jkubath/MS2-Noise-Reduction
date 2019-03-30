# protein.py is a helper file for noise_reduction_ms2.py
#
# Description: The main() function is used to take a MS2
# format data file, read the spectrum information (m/z and intensity)
# and write the values binned to the nearest integer value of the m/z
# value
import numpy as np # zero out array
import matplotlib.pyplot as plt # show ms2 data
import csv # write output as csv
import os # get current working directory for output

# writes the data list in a csv format to the file
def outputData(writer, data, index, length):
	# Error checking for empty data
	if len(data) == 0 or len(data[index]) == 0:
		print("Data is empty. Index: {}".format(index))
		return -1
	if not writer:
		print("Ouput file object is null")
		return -1

	# write the data to file
	i = 0
	string = []
	while i < length-1:
		string.append(str(int(data[index][i])))
		string.append(",")
		i += 1

	# replace last ',' with a newline
	string[-1] = "\n"

	writer.write("".join(string))

# Read the MS2 format data and saves the spectrum data to the outputFileObject
# Returns a numpy array of the data
def readFile(fileObject, arraySize, outputFileObject, writeToFile = False):
	data = [] # list to hold the list of spectrum data

	if not writeToFile:
		print("Not writing to file")

	spectrum_count = 0
	addIndex = 0
	firstIteration = True
	# read all the data in the file
	for line in fileObject:
		# hold the peak data
		splitLine = line.split(" ")

		# skip the header information on the first run
		# otherwise output the data we read
		if(splitLine[0][0] == 'H' or splitLine[0][0] == 'S'):
			continue
		elif splitLine[0][0] == 'Z':
			# first iteration through, we have not read any spectrum data yet
			if firstIteration:
				firstIteration = False
				data.append(np.zeros(arraySize))
				continue
			# finished reading a spectrum, output to file
			else:
				if writeToFile:
					if outputData(outputFileObject, data, spectrum_count, arraySize) == -1:
						print("Error printing data")
						break
				addIndex = 0
				spectrum_count += 1
				data.append(np.zeros(arraySize))
				if spectrum_count % 100000 == 0:
					print("Wrote {} peptides".format(spectrum_count))
				continue
		elif splitLine[0][0] == "\n":
			continue

		# Add the peak point to the correct bin
		try:
			# change the index
			if int(round(float(splitLine[0]))) > addIndex:
				addIndex = int(round(float(splitLine[0])))

			# move down the array until an unfilled index is found
			# if overlapping data points are found, keep the higher
			# intensity peak
			if float(splitLine[1]) > data[spectrum_count][addIndex]:
				data[spectrum_count][addIndex] = int(float(splitLine[1]))

			# print(addIndex, data[addIndex])


		# header information
		except ValueError:
			print("Spectrum count: {}".format(spectrum_count))
			print("First char: {}".format(splitLine[0][0]))

	return np.array(data)


# Read the MS2 format data and saves the spectrum data to the outputFileObject
def readFileNoReturn(fileObject, arraySize, outputFileObject, writeToFile = True):
	data = [] # list to hold the list of spectrum data

	if not writeToFile:
		print("Not writing to file")

	spectrum_count = 0
	addIndex = 0
	firstIteration = True
	# read all the data in the file
	for line in fileObject:
		# hold the peak data
		splitLine = line.split(" ")

		# skip the header information on the first run
		# otherwise output the data we read
		if(splitLine[0][0] == 'H' or splitLine[0][0] == 'S'):
			continue
		elif splitLine[0][0] == 'Z':
			# first iteration through, we have not read any spectrum data yet
			if firstIteration:
				firstIteration = False
				data = []
				continue
			# finished reading a spectrum, output to file
			else:
				if writeToFile:
					while(addIndex < arraySize):
						data.append("0")
						addIndex += 1
					string = ""
					string = ",".join(data)
					string += "\n"
					# print(string)
					outputFileObject.write(string)
					# if outputData(outputFileObject, data, spectrum_count, arraySize) == -1:
					# 	print("Error printing data")
					# 	break
				addIndex = 0
				spectrum_count += 1
				data = []
				if spectrum_count % 100000 == 0:
					print("Wrote {} peptides".format(spectrum_count))
				continue
		elif splitLine[0][0] == "\n":
			continue

		# Add the peak point to the correct bin
		try:
			# change the index
			while(int(round(float(splitLine[0]))) >= addIndex):
				data.append("0")
				addIndex += 1
				# addIndex = int(round(float(splitLine[0])))

			# move down the array until an unfilled index is found
			# if overlapping data points are found, keep the higher
			# intensity peak
			if float(splitLine[1]) > float(data[addIndex-1]):
				data[addIndex-1] = str(int(float(splitLine[1])))

			# print(addIndex, data[addIndex])


		# header information
		except ValueError:
			print("Spectrum count: {}".format(spectrum_count))
			print("First char: {}".format(splitLine[0][0]))

	return np.array(data)

# python script to read MS2 data, bin spectra values to nearest m/z
# integer value, and output in csv format
def main():
	print("Protein script")

	# Default variables
	#filePath = "/Users/jonah/Desktop/research/"
	filePath = str(os.getcwd()) + "/large/"
	# filePath = "D:/MS2/"
	# fileName = "train_noise.ms2"
	# fileOutput = "train_noise_binned.ms2"

	fileName = "test_no_noise.ms2"
	fileOutput = "test_no_noise_binned.ms2"

	# hold the peak information
	dataLength = 7000

	# create the output directory if it doesn't exist
	os.makedirs(filePath, exist_ok=True)

	# open file for reading
	try:
		inputFileObject = open(filePath + fileName, "r")
		# write to the output file
		outputFileObject = open(filePath + fileOutput, "w")
	except (OSError, IOError) as e:
		print(e)
		return

	# read the data
	# data = readFile(inputFileObject, dataLength, outputFileObject, True)
	readFileNoReturn(inputFileObject, dataLength, outputFileObject, True)

if __name__ == '__main__':
	main()

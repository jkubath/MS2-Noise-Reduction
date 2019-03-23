import numpy as np # zero out array
import matplotlib.pyplot as plt # show ms2 data
import csv # write output as csv

# writes the data in a csv format to the file
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
	while i < length-1:
		writer.write(str(int(data[index][i])) + ",")
		i += 1

	writer.write(str(data[index][i]) + "\n")

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
				if spectrum_count % 1000 == 0:
					print("Wrote {} peptides".format(spectrum_count))
				continue

		# Add the peak point to the correct bin
		try:
			# change the index
			if int(float(splitLine[0])) > addIndex:
				addIndex = int(float(splitLine[0]))

			# move down the array until an unfilled index is found
			# while(data[addIndex] != 0):
			# 	addIndex += 1
			if float(splitLine[1]) > data[spectrum_count][addIndex]:
				data[spectrum_count][addIndex] = int(float(splitLine[1]))

			# print(addIndex, data[addIndex])


		# header information
		except ValueError:
			print("Error at {}".format(splitLine[0]))
			print("First char: {}".format(splitLine[0][0]))

	return np.array(data)

# python script for protein research
def main():
	print("Protein script")

	# Default variables
	#filePath = "/Users/jonah/Desktop/research/"
	filePath = "C:/Users/koob8/Desktop/nn/output/"
	fileName = "noNoise.ms2"
	#fileName = "sampleOut.ms2"
	fileOutput = "peptides_sd.ms2"

	# hold the peak information
	dataLength = 5000

	# open file for reading
	try:
		inputFileObject = open(filePath + fileName, "r")
		# write to the output file
		outputFileObject = open(filePath + fileOutput, "w")
	except (OSError, IOError) as e:
		print(e)
		return

	# read the data
	data = readFile(inputFileObject, dataLength, outputFileObject, True)

if __name__ == '__main__':
	main()

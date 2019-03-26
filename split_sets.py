import os
import random


def generateSets(folder = "/output"):
    # File names
    #-------------------------------------------------------------------------
    # no_noise_file = "no_noise.ms2" # ms2 file containing with no noise
    # noise_file = "noise.ms2" # regular ms2 data
    #
    # no_noise_output_file = "no_noise_binned.ms2"
    # noise_output_file = "noise_binned.ms2"
    #
    #
    inputFile = "peptides_5000.txt"

    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"

    train_min = 30
    valid_min = 10

    write_noise_output = True
    write_no_noise_output = True

    # Create file readers for training set
    #-------------------------------------------------------------------------
    print("Creating training file objects")
    try:
    	# labels
        peptides_object = open(folder + inputFile, "r")

    	# write to the output files
        train_object = open(folder + train_file, "w")
        valid_object = open(folder + valid_file, "w")
        test_object = open(folder + test_file, "w")
    except (OSError, IOError) as e:
    	print(e)
    	exit()

    count = 0
    for line in peptides_object:
        # random integer to decide which file to write to
        randomChoice = random.randint(0, 101)

        if randomChoice > train_min:
            train_object.write(line)
        elif randomChoice > valid_min:
            valid_object.write(line)
        else:
            test_object.write(line)

        if count % 1000 == 0:
            print("Wrote {} peptides".format(count))

        count += 1

def main():
    print("Splitting data sets")

    # Read in the data
    folder = "/Users/jonah/Desktop/MS2-Noise-Reduction/"
    outputFolder = "output1/"

    noNoise = "no_noise.ms2"
    noise = "noise.ms2"

    os.makedirs(outputFolder, exist_ok=True)

    generateSets(folder + outputFolder)


if __name__ == '__main__':
    main()

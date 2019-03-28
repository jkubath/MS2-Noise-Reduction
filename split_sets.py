# split_sets.py is used to break apart a file containing peptide strings
# The input file will be read and a random integer is chosen.  This variable
# decides which file to output the peptide string to (test, train, or validate)
import os # operating system directory structure
import random # generating random integer to decide which file (test, train, validate) to write the peptide to


def generateSets(inputFile, inputFolder, outputFolder = "/output"):
    # File names for output
    #-------------------------------------------------------------------------
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"

    # percentages to write data to files
    # train_min = 31-100 get written to training file
    # valid_min = 11 - 30 get written to validation file
    # The rest are written to test file
    train_min = 30
    valid_min = 10

    # maximum number of peptides to use
    max_peptides = 1000

    # Create file readers for training set
    #-------------------------------------------------------------------------
    print("Creating file objects")
    try:
    	# read peptide sequences
        peptides_object = open(inputFolder + inputFile, "r")

    	# write to the output files
        train_object = open(outputFolder + train_file, "w")
        valid_object = open(outputFolder + valid_file, "w")
        test_object = open(outputFolder + test_file, "w")
    except (OSError, IOError) as e:
    	print(e)
    	exit()

    count = 0
    for line in peptides_object:
        # random integer to decide which file to write to
        randomChoice = random.randint(1, 100)

        if randomChoice > train_min:
            train_object.write(line) # write to training file
        elif randomChoice > valid_min:
            valid_object.write(line) # write to validation file
        else:
            test_object.write(line) # write to test file

        # print status
        if count % 1000000 == 0:
            print("Wrote {} peptides".format(count))

        count += 1

        if count == max_peptides:
            break

    print("Wrote {} peptides".format(count))

def main():
    print("Splitting data sets")

    # Determine working directory and output folder
    folder = str(os.getcwd())
    inputFolder = "/protein/"
    outputFolder = "/peptide/"

    # File containing a peptide sequences (one sequence per line)
    inputFile = "peptideData.txt"

    print("Output folder: {}".format(folder + outputFolder))

    # make the output folder if it doesn't exist
    os.makedirs(folder + outputFolder, exist_ok=True)

    # read the input peptide file and split the peptides between test, train,
    # and validation files
    generateSets(inputFile, folder + inputFolder, folder + outputFolder)


if __name__ == '__main__':
    main()

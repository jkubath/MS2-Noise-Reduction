# gen_peptide.py
# 1. fasta file is read and protein id, protein sequence
# 	is written to file.
# 2. protein file is then read and all the peptide sequences
# 	are generated and written to file.
# These substrings are limited to specific lengths (6 - 50)
from Bio import SeqIO # used to read fasta file
import os # used to access operating system directory structure

# Set of peptides cleaved from the various proteins
peptide_data = {}

# Read the given fasta file
def readFasta(fastaPath, outputFilePath = "C:/", writeOutput = False):
    # open the fasta for reading
    fileObject = open(fastaPath, "r")

    fastaData = SeqIO.parse(fileObject, 'fasta')

    if writeOutput:
        outputFileObject = open(outputFilePath, "w")

        for fasta in fastaData:
            string = "{},{}\n".format(fasta.id, fasta.seq)
            outputFileObject.write(string)

# Break the string into all the various substrings (peptides)
def cleaveProtein(proteinSequence, minLength, maxLength):
    peptides = []

    for i in range(len(proteinSequence)):
        for j in range(minLength, maxLength):
            if j + i >= len(proteinSequence):
                break
            # print("peptide: {}".format(proteinSequence[i:j]))
            # get all the substrings
            if not j - i < 6:
                peptides.append(proteinSequence[i:j])

    return peptides

# call cleaveProtein on all the proteins.  This will generate all the peptide
# sequences and save them to peptide_data
def generatePeptide(proteinOutputFile, minLen = 6, maxLen = 50):
    global peptide_data

    # open the fasta for reading
    fileObject = open(proteinOutputFile, "r")

    proteinTotal = 0
    peptideTotal = 0

    for line in fileObject:
        splitLine = line.split(",")
        id = splitLine[0]
        sequence = splitLine[1]

        # print("Protein id: {}".format(id))
        # print("Sequence: {}".format(sequence))

        proteinTotal += 1

        cleaved_peptides = cleaveProtein(sequence, minLen, maxLen)

        peptideTotal += len(cleaved_peptides)

        for peptide in cleaved_peptides:
            # skip if the sequence has already been found
            if peptide in peptide_data:
                continue
            else:
                peptide_data[peptide] = [0]
                continue

        if proteinTotal % 100 == 0:
            print("Read {} proteins".format(proteinTotal))

# Write the peptide_data to the peptideOutputFile
def writePeptideData(peptideOutputFile):
    global peptide_data

    outputObject = open(peptideOutputFile, "w")

    string = ""
    for peptide, charges in peptide_data.items():
        outputObject.write(peptide + "\n")

def main():
    print("Generating peptide data from fasta")

    # Read in the data
    outputFolder = str(os.getcwd()) + "/protein/"

    os.makedirs(outputFolder, exist_ok=True)

    proteinOutputFile = outputFolder + "proteinData.txt" # protein id, protein sequence
    peptideOutputFile = outputFolder + "peptideData.txt" # strings of peptide sequences
    fastaFile = "yeast.fasta"

    # Read fasta file with SeqIO api
    readFasta(fastaFile, proteinOutputFile, True)

    # generate peptides (every substring of length 6 to 50)
    generatePeptide(proteinOutputFile, 6, 50)

    # write strings to output file
    writePeptideData(peptideOutputFile)


if __name__ == '__main__':
    main()

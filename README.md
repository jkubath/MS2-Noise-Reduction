# MS2-Noise-Reduction
Implementing neural networks to reduce noise in MS2 spectra.

Dir: MaSS-Simulator
Description: software to generate MS2 spectrum data.
Source: https://github.com/pcdslab/MaSS-Simulator

Dir: MNIST_data
Description: Image data downloaded when running noise_reduction_tutorial.py.  Images are hand written numbers 0 to 9.

Dir: small_output
  Description: Folder containing input, validation, and test MS2 data for noise_reduction_ms2.py.  This data was generated with the MaSS-Simulator software.
    no_noise: Files without the added noise peaks
    binned: Files output from protein.py that have the generated MS2 data binned.  The m/z values are binned to integer values instead of float values.

Dir: big_data
  Description: Folder containing the same information as small_output, but the size of the data was started at 5000 peptides.  This data was then split randomly into train, validation, and test data sets.


General Outline:
  1. Start with a fasta protein database
  2. Call gen_peptide.py to read the fasta file, output proteins in
    a protein id, protein sequence format.  This python script will then
    read the protein file and generate all the peptide sequences that can be
    generated for the protein file.  Our implementation limits sequences to
    lengths of 6 - 50. (data written to "protein" directory)
  3. Call split_sets.py to read the peptide data and split the peptide sequences
    between train, test, and validation files.
  4. Call MaSS-Simulator/SimSpec.java to generate the theoretical MS2 spectra
    for the 3 peptide files (train, test, and validation).  
    Compilation
    ----------------------------------------------------------------------------
    There is a compile_java.txt in the src directory of MaSS-Simulator with instructions
    on compiling the software

    cd "MaSS-Simulator/src"
    javac -cp "corba-api-5.0.1.jar" SimSpec.java Ion.java MyPair.java

    Input
    ----------------------------------------------------------------------------
    Input files are included in the src/data folder
    peptides_5000.txt: List of peptide sequences
      The included file is ~5000 peptide sequences from yeast.fasta
    params.txt: default parameters
    mods.ptm: modifications to amino acids (none are default)

    Run
    ----------------------------------------------------------------------------
    java SimSpec 1 1 ./data/peptides_5000.txt noise.ms2 ./data/params.txt ./data/mods.ptm

    Output
    ----------------------------------------------------------------------------
    This java file will output two files (no_noise.ms2 and noise.ms2).  
      noise.ms2 is the input data that contains the b-ion, y-ion, and noisy peaks of the input peptide sequences.
      no_noise.ms2 contains only the b-ion and y-ion data.  This is used as label data for noise.ms2.

    *** The output files must be saved to the expected folder in noise_reduction_ms2.py ***
    This folder is defaulted to "MS2-Noise-Reduction/big_data" and named appropriately

    Train files:
      no_noise.ms2
      noise.ms2
    Validation files:
      valid_no_noise.ms2
      valid_noise.ms2
    Test files:
      test_noise.ms2
      test_no_noise.ms2


  5. Call noise_reduction_ms2.py
    A. Read the 6 MS2 files
    B. Bin the m/z float values to nearest integer and write to binned files
    C. build a neural network to learn how to reduce the noise in the MS2 data
    D. test

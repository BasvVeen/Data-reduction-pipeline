# Data-reduction-pipeline

In this github depository, all the necessary files are given to reduce an image. The jupyter notebooks are the files with which you can reduce your image. Within these notebooks commentary is given to explain the necessary steps to run the pipeline.

Version 0.9.1 (newest version of pynpoint) does not work with some of the functions I am using and I have therefore opted to use 0.8.1, as recommended by Alexander Bohn (creator of these functions). This will cause a couple of errors, namely : that the datatype it gets is inttype32 instead of inttype64. This error will be present in each of the functions used in the pipeline. To fix this you need to go to the file in the PynPoint package installed and add the following line under the line where MEMORY is imported: memory = np.int64(memory), I have not found a way around this, so this needs to be changed by hand. 

The files input files need to be .fits files downloaded from the ESO archive, with their corresponding calibration files (DARK, FLAT, SKY, CENTER), all sorted in a folder called 'input' by their type (so a separate folder for each of the type of images; SCIENCE, DARK, FLAT, SKY, CENTER). I did this sorting by hand.

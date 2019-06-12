Image Processing for Computer Vision using CUDA

This project explores how image convolution operation can be parallelized using a GPU and achieve increasingly higher speedups when
compared to a CPU. The following sections contain the file organization and procedure to run the code.

File Organization
=================

The project folder ImageConvolutionUsingCUDA contains the following subfolders:-

1) headers -- Contains all the header files necessary to run the code.

======================================================================================================================================
NOTE: All convolution code 2), 3), 4), 5) have their code set to take in a 1024x1024 size image. To use other sizes, a line in the code
must be changed to take in a particular image size. That line in the code differs for all the codes. 
======================================================================================================================================

2) sequential -- Contains the code for sequential convolution. convSeq.cu & convSeq_5x5.cu for 3x3 and 5x5 filter kernels respectively. batch_runs.sh and batch_runs_5x5.sh are shell scripts(for 3x3 nad 5x5 kernels) to run the convolution 20 times and average the total computation time. reports folder contains the all the outputs from running the batch_runs script for different image and filter sizes.

3) globalMemory -- Contains the code for convolution using Global Memory. convGlobalMem.cu & convGlobalMem_5x5.cu for 3x3 and 5x5 filter kernels respectively. batch_runs.sh and batch_runs_5x5.sh are shell scripts to run the convolution 20 times and average the total computation time. reports folder contains the all the outputs from running the batch_runs script for different image and filter sizes.


4) constantMemory -- Contains the code for convolution using Constant Memory. convConstMem.cu & convConstMem_5x5.cu for 3x3 and 5x5 filter kernels respectively. batch_runs.sh and batch_runs_5x5.sh are shell scripts to run the convolution 20 times and average the total computation time. reports folder contains the all the outputs from running the batch_runs script for different image and filter sizes.

5) sharedMemory -- Contains the code for convolution using Global Memory. convSharedMem.cu & convSharedMem_5x5.cu for 3x3 and 5x5 filter kernels respectively. batch_runs.sh and batch_runs_5x5.sh are shell scripts to run the convolution 20 times and average the total computation time. reports folder contains the all the outputs from running the batch_runs script for different image and filter sizes.

5) images -- Contains all the raw images and the convoluted images(outputs).
		Raw images - city.jpg, landscape1024.jpg, lena512.jpg, cameraman256.jpg
		Convoluted image - all images apart from the raw images are outputs, dumped from different convolution algorithms

6) videos -- Contains two folders, 1) gtav_FINAL - Contains a videostream converted into frames.
				   2) gtav_OUT - Contains the video frames after convolution.


Procedure to run DEMO
====================
1) The DEMO folder constains the necessary files to run a sequential convolution and a global memory convolution. NOTE: the header file paths inside convSeq.cu and convGlobalMemory.cu might have to be changed to refer to the file inside the headers folder inside DEMO. 

2) After requesting a GPU node in Comet, a job can be run using the `source batch_runs.sh //home/ksugumar/ImageConvolutionUsingCUDA/DEMO/images/city.jpg //home/ksugumar/ImageConvolutionUsingCUDA/DEMO/city_out.jpg`. NOTE: the arguments are paths to the input and output iamges for performing the operation


Procedure to run scripts
========================
To run any script, follow the steps below:-
The following example shows how to run the global memory convolution.

1) Request a GPU node using the command:  
srun --partition=gpu-shared --nodes=1 --ntasks-per-node=6 --gres=gpu:1 --pty -t 00:30:00 --wait=0 /bin/bash

2)run `module load cuda` to load the nvcc cuda compiler

3) Inside globalMemory folder, run `source batch_runs.sh` while giving 2 arguments to it. (i) the path to the input image and (ii) the path to the output image.

	example: source batch_runs.sh //home/ksugumar/ImageConvolutionUsingCUDA/images/city.jpg //home/ksugumar/ImageConvolutionUsingCUDA/image/city_out.jpg

This will dump the convoluted image using the 3x3 outline filter. run batch_runs_5x5.sh to use the 5x5 gaussian blur filter.

4) For running convolution on video frames, go to sharedMemory folder. run `source run_video.sh` with the arguemnts (i) path to folder containing the video frames and (ii) path to folder to store the output frames.


ALTERNATIVELY
+++++++++++++
Any of the above discussed codes can also be run by directly compiling the code and calling it with the necessary arguments
After requesting a GPU node and loading the cuda module, compile the code using `nvcc -o convSeq convSeq.cu` command. Then run convSeq executable file by `./convSeq "path/to/input/image" "path/to/output/image"`.

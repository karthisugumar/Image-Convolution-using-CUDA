#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include </home/ksugumar/project/headers/helper_functions.h>
#include </home/ksugumar/project/headers/helper_cuda.h>
#include "device_launch_parameters.h"
#include <chrono>

#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/home/ksugumar/project/headers/stb_image.h"
#include "/home/ksugumar/project/headers/stb_image_write.h"

using namespace std;
using namespace chrono;

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(int argc, char** argv)
{
	// Read the image file on host
	int rows, cols, bpp;
	uint8_t* h_original_image = stbi_load(argv[1], &cols, &rows, &bpp, 1);
	
	// Declare image variables
	int padding_size = 2;
	unsigned int r_pad = rows + 2 * padding_size;
	unsigned int c_pad = cols + 2 * padding_size;
	int imsize = rows*cols;
	int imsize_pad = r_pad*c_pad;

	//Allocate space on host for padded input image
	float **h_padded_image;
	h_padded_image = new float*[r_pad];
	for (int i = 0; i < r_pad; i++)
	{
		h_padded_image[i] = new float[c_pad];
	}

	// Fill the 2D array with zeros
	for (int i = 0; i < r_pad; i++)
	{
		for (int j = 0; j < c_pad; j++)
		{
			h_padded_image[i][j] = 0;
		}
	}

	// Copy pixels from the original image to the 2D array, without affecting the padded 0
	for (int i = padding_size; i < r_pad - padding_size; i++)
	{
		for (int j = padding_size; j < c_pad - padding_size; j++)
		{
			h_padded_image[i][j] = *(h_original_image + ((i - padding_size)*cols) + (j - padding_size));
		}
	}
	
	// Convert the padded image to a 1D array. Accessing 1D arrays are more efficient in GPUs
	float *h_padded_image_1d = new float[imsize_pad];
	for (int q = 0; q < r_pad; q++)
	{
		for (int t = 0; t < c_pad; t++)
		{
			h_padded_image_1d[q * r_pad + t] = h_padded_image[q][t];
		}
	}

	// delete the original 2D padded image after reshaping it to 1D
	delete h_padded_image;
	
	
	// Initialize the kernel to be used for convolution as a 1D array
	
	// Gaussian blur filter 5x5
	float h_filter[25] = { 1, 4,  6,  4, 1, \
		4, 16, 24, 16, 4, \
		6, 24, 36, 24, 6, \
		4, 16, 24, 16, 4, \
		1, 4,  6,  4, 1, };

	for (int f = 0; f < 25; f++) {
		h_filter[f] /= 256.0;
	}

//	float h_filter[9] = { -1, -1, -1, \
//			     - 1,  8, -1, \
//			     - 1, -1, -1 };
//
	// Sharpen
	//float h_filter[9] = { 0, -1,  0, \
	//			-1,  5, -1, \
	//			0, -1,  0  };

	// Blur
	//double h_filter[9] = { 0.0625, 0.125,  0.0625, \
		//		 0.125, 0.125, 0.125, \
		//		 0.0625, 0.125, 0.0625 };

	// Sobel (Left, Right, Top, Bottom) Shown below is Left Sobel
	//float h_filter[9] = { 1, 0, -1, \
		//		  2, 0, -2, \
		//	      1, 0, -1 };


	// Emboss 
	//float h_filter[9] = { -2, -1, 0, \
		//	  -1,  1, 1, \
		//	   0,  1, 2 };


	// Identity
	//float h_filter[9] = { 0, 0, 0, \
		//		0, 1, 0, \
		//		0, 0, 0  };

	// Initialize a 1D array to hold the convoluted image
	float *h_conv_image_1d = new float[imsize];

	unsigned int filter_size = 2 * padding_size + 1;

	
// ***** CONVOLUTION STARTS HERE ! *****

	auto h_start = steady_clock::now();

	int conv_count = 0;
	// The loops for the pixel coordinates
	for (int i = padding_size; i < r_pad - padding_size; i++) {
		for (int j = padding_size; j < c_pad - padding_size; j++) {

			// The multiply-add operation for the pixel coordinate ( j, i )
			unsigned int oPixelPos = (i - padding_size) * cols + (j - padding_size);
			h_conv_image_1d[oPixelPos] = 0;
			for (int k = -padding_size; k <= padding_size; k++) {
				for (int l = -padding_size; l <= padding_size; l++) {
					unsigned int iPixelPos = (i + k) * c_pad + (j + l);
					unsigned int coefPos = (k + padding_size) * filter_size + (l + padding_size);
					h_conv_image_1d[oPixelPos] += h_padded_image_1d[iPixelPos] * h_filter[coefPos];
				}
			}
			conv_count += 1;
		}
	}

	auto h_end = steady_clock::now();
	cout << "\nConvolution completed!\n";

// ***** CONVOLUTION ENDS HERE ! *****

	cout << "Total Elapsed time: " << (duration<double>\
					(h_end - h_start).count())*1000.0 << " ms\n" << endl;
	
	static uint8_t conv_image_final[1024][1024];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			conv_image_final[i][j] = 0;
		}
	}

	// perform convertion of 1d to 2d
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int pixel = h_conv_image_1d[i*rows + j];
			if (pixel > 255)
				conv_image_final[i][j] = 255;
			else if (pixel < 0)
				conv_image_final[i][j] = 0;
			else
				conv_image_final[i][j] = pixel;

			//	cout << pixel << " ";
		}
	}

	// Write convoluted image to file (.jpg)
	stbi_write_jpg(argv[2], cols, rows, 1, conv_image_final, cols);
	
	return 0;
}

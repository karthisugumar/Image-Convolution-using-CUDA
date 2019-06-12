#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include </home/ksugumar/project/headers/helper_functions.h>
#include </home/ksugumar/project/headers/helper_cuda.h>
#include "device_launch_parameters.h"
#include <chrono>

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

//Kernel function for image filtering using convolution
__global__ void imageFilteringkernel\
	(const float *d_image_pad, const unsigned int r_pad, const unsigned int c_pad, const float *d_filter,
	const int padding_size, float *d_conv_image, const unsigned int rows, const unsigned int cols)

{
	unsigned int filter_size = 2 * padding_size + 1;

	//Determine the pixel co-ordinates
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + padding_size;
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + padding_size;

	//Multiply and add operation for pixel (j,i)
	if (j > padding_size && j < c_pad - padding_size && i > padding_size && i < r_pad - padding_size)
	{
		unsigned int conv_pixel_pos = (i - padding_size) * cols + (j - padding_size);
		d_conv_image[conv_pixel_pos] = 0.0;
		for (int k = -padding_size; k <= padding_size; k++)
		{
			for (int l = -padding_size; l <= padding_size; l++)
			{
				unsigned int img_pixel_pos = (i + k) * c_pad + (j + l);
				unsigned int filter_pos = (k + padding_size) * filter_size + (l + padding_size);
				d_conv_image[conv_pixel_pos] += d_image_pad[img_pixel_pos] * d_filter[filter_pos];
			}
		}
	}
}

inline unsigned int iDivUp(const unsigned int &a, const unsigned int &b) { return (a%b != 0) ? (a / b + 1) : (a / b); }

int main(int argc, char** argv)
{
	// Read the image file on host
	int rows, cols, bpp;
	uint8_t* h_original_image = stbi_load(argv[1], &cols, &rows, &bpp, 1);
	cout << "The number of rows is " << rows << "\n";
	cout << "The number of columns is " << cols << "\n";
	
	// Declare Image variables
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
	//float h_filter[9] = {0, 0, 0, \
	//	0,  1, 0, \
	//	0, 0, 0 };

	// Initialize a 1D array to hold the convoluted image
	float *h_conv_image_1d = new float[imsize];

	unsigned int filter_size = 2 * padding_size + 1;

// MEMORY ALLOCATION ON DEVICE STARTS HERE

	//Allocate memory on device for image and transfer image from host to device
	float *d_padded_image;
	unsigned int d_imsize_pad = r_pad * c_pad * sizeof(float);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_padded_image), d_imsize_pad));
	

	auto h_start = steady_clock::now();
	checkCudaErrors(cudaMemcpy(d_padded_image, h_padded_image_1d, d_imsize_pad, cudaMemcpyHostToDevice));

	//Allocate memory on device for filter and transfer filter from host to device
	float *d_filter;
	unsigned int d_filtersize = filter_size * filter_size * sizeof(float);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_filter), d_filtersize));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, d_filtersize, cudaMemcpyHostToDevice));

	//Set up the grid and block dimensions for execution
	const unsigned int block_col = 32;
	const unsigned int block_row = 32;
	const dim3 grid( iDivUp(cols, block_col), iDivUp(rows, block_row));
	const dim3 threadBlock(block_col, block_row);

	//Memory allocation for filtered image
	float *d_conv_image;
	unsigned int conv_imsize = rows * cols * sizeof(float);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_conv_image), conv_imsize));


// **** CONVOLUTION STARTS HERE ! ****
	
	float elapsed = 0;
	cudaEvent_t start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	checkCudaErrors(cudaDeviceSynchronize());
	imageFilteringkernel <<<grid, threadBlock>>>(d_padded_image, r_pad, c_pad, d_filter, padding_size, d_conv_image, rows, cols);
	checkCudaErrors(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	cout << "Total Elapsed Time for the Kernel(GPU): " << elapsed << " ms" << endl;

	checkCudaErrors(cudaMemcpy(h_conv_image_1d, d_conv_image, conv_imsize, cudaMemcpyDeviceToHost));

	auto h_end = steady_clock::now();
	
	cout << "Total Elapsed Time(including data transfer): " << (duration<double>\
					(h_end - h_start).count())*1000.0 << " ms\n" << endl;

	
// **** CONVOLUTION ENDS HERE ! ****
	
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

		}
	}

	// Write convoluted image to file
	stbi_write_jpg(argv[2], cols, rows, 1, conv_image_final, cols);

	return 0;
}

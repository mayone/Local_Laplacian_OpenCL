#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <sys/time.h>
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define PNG_DEBUG 3
#include <png.h>

#define NUM_KERNELS 8
#define GEN_FLOATING 0
#define GEN_GRAY 1
#define GEN_GPYRAMID0 2
#define DOWNSAMPLE_KERNEL 3
//#define GEN_LPYRAMID 4
#define GEN_OUTLPYRAMIDLOWEST 4
#define GEN_OUTLPYRAMID 5
#define GEN_OUTGPYRAMID 6
#define GEN_OUTPUT 7

#define maxJ 8
#define levels 8

void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

int x, y;

int width, height;
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;

void read_png_file(char* file_name);
void write_png_file(char* file_name);
void process_file(void);

void getRGB(uint8_t *r, uint8_t *g, uint8_t *b)
{
	for (y = 0; y < height; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < width; x++) {
			png_byte* ptr = &(row[x*4]);
			r[width*y + x] = ptr[0];
			g[width*y + x] = ptr[1];
			b[width*y + x] = ptr[2];
		}
	}
}

void returnRGB(uint8_t *r, uint8_t *g, uint8_t *b)
{
	for (y = 0; y < height; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < width; x++) {
			png_byte* ptr = &(row[x*4]);
			ptr[0] = r[width*y + x];
			ptr[1] = g[width*y + x];
			ptr[2] = b[width*y + x];
		}
	}
}

cl_program load_program(cl_context context, cl_device_id device, const char* filename);

int clCreateKernels(cl_program program, cl_kernel **kernels_ptr);
int clReleaseKernels(cl_kernel *kernels);

int main(int argc, char **argv)
{
	uint8_t *src_r;
	uint8_t *src_g;
	uint8_t *src_b;

	uint8_t *dst_r;
	uint8_t *dst_g;
	uint8_t *dst_b;

	// OpenCL
	cl_platform_id *platforms;
	cl_context context;
	cl_uint num_devices;
	cl_device_id *devices;
	cl_command_queue queue;
	cl_program program;
	cl_kernel *kernels;
	char *devName;
	char *devVer;
	size_t cb;
	cl_int err;
	cl_uint num;
	size_t global_work_size[2];
	size_t local_work_size[2] = {16, 16};
	struct timeval tim;   
	
	// get the id of supporting OpenCL platforms
	err = clGetPlatformIDs(0, 0, &num);
	if (err != CL_SUCCESS)
	{
		perror("Unable to get platforms");
		return 0;
	}
	platforms = (cl_platform_id*)malloc(num * sizeof(cl_platform_id));
	clGetPlatformIDs(num, &platforms[0], NULL);
	printf("There are %d platform(s) on this device\n", num);
	
	// create a OpenCL context
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
	context = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
	if (context == 0)
	{
		perror("Can't create OpenCL context");
		return 0;
	}

	// get a list of devices
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	devices = (cl_device_id*) malloc(cb / sizeof(cl_device_id)); 
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

	// get the number of devices
	clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, 0, NULL, &cb);
	clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, cb, &num_devices, 0);
	printf("There are %d device(s) in the context\n", num_devices);

	// show devices info
	for(int i = 0; i < num_devices; i++)
	{
		// get device name
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &cb);
		devName = (char*) malloc(sizeof(char) * cb);
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, cb, &devName[0], NULL);
		devName[cb] = 0;
		printf("Device: %s", devName);
		free(devName);
		
		// get device supports version
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 0, NULL, &cb);
		devVer = (char*) malloc(sizeof(char) * cb);
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, cb, &devVer[0], NULL);
		devVer[cb] = 0;
		printf(" ( supports %s)\n", devVer);
		free(devVer);
	}

	// construct command queue
	queue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (queue == 0)
	{
		perror("Can't create command queue\n");
		clReleaseContext(context);
		return 0;
	}

	// create and compile the program object
	program = load_program(context, devices[0], "local_laplacian.cl");
	if (program == 0)
	{
		perror("Error, can't load or build program\n");
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}

	// create kernel objects from program
	err = clCreateKernels(program, &kernels);
	assert(err == CL_SUCCESS);

	if (argc != 3)
		abort_("Usage: program_name <file_in> <file_out>");

	read_png_file(argv[1]);

	gettimeofday(&tim, NULL);  
    double dTime1 = tim.tv_sec+(tim.tv_usec/1000000.0); 

	src_r = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	src_g = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	src_b = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	dst_r = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	dst_g = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	dst_b = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	// create cl buffers
	cl_mem src_r_d = clCreateBuffer(context, 0, sizeof(uint8_t) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem src_g_d = clCreateBuffer(context, 0, sizeof(uint8_t) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem src_b_d = clCreateBuffer(context, 0, sizeof(uint8_t) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);

	cl_mem floating_r = clCreateBuffer(context, 0, sizeof(float) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem floating_g = clCreateBuffer(context, 0, sizeof(float) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem floating_b = clCreateBuffer(context, 0, sizeof(float) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);

	cl_mem gray = clCreateBuffer(context, 0, sizeof(float) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);

	cl_mem gPyramid[maxJ][levels];
	for (int j = 0; j < maxJ; j++) {
		for (int k = 0; k < levels; k++) {
			gPyramid[j][k] = 
				clCreateBuffer(context, 0, 
					sizeof(float) * (width >> j) * (height >> j),
					NULL, &err);
			assert(err == CL_SUCCESS);
		}
	}
	cl_mem inGPyramid[maxJ];
	for (int j = 0; j < maxJ; j++) {
		inGPyramid[j] = 
			clCreateBuffer(context, 0, 
				sizeof(float) * (width >> j) * (height >> j),
				NULL, &err);
		assert(err == CL_SUCCESS);
	}
	cl_mem outLPyramid[maxJ];
	for (int j = 0; j < maxJ; j++) {
		outLPyramid[j] = 
			clCreateBuffer(context, 0, 
				sizeof(float) * (width >> j) * (height >> j),
				NULL, &err);
		assert(err == CL_SUCCESS);
	}
	cl_mem outGPyramid[maxJ];
	for (int j = 0; j < maxJ; j++) {
		outGPyramid[j] = 
			clCreateBuffer(context, 0, 
				sizeof(float) * (width >> j) * (height >> j),
				NULL, &err);
		assert(err == CL_SUCCESS);
	}
	cl_mem dst_r_d = clCreateBuffer(context, 0, sizeof(uint8_t) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem dst_g_d = clCreateBuffer(context, 0, sizeof(uint8_t) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);
	cl_mem dst_b_d = clCreateBuffer(context, 0, sizeof(uint8_t) * width * height, NULL, &err);
	assert(err == CL_SUCCESS);


	getRGB(src_r, src_g, src_b);

	err = clEnqueueWriteBuffer(queue, src_r_d, CL_TRUE, 0, sizeof(uint8_t) * width * height, src_r, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	err = clEnqueueWriteBuffer(queue, src_g_d, CL_TRUE, 0, sizeof(uint8_t) * width * height, src_g, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	err = clEnqueueWriteBuffer(queue, src_b_d, CL_TRUE, 0, sizeof(uint8_t) * width * height, src_b, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	
	global_work_size[0] = width;
	global_work_size[1] = height;

	// Floating
	clSetKernelArg(kernels[GEN_FLOATING], 0, sizeof(floating_r), &floating_r);
	clSetKernelArg(kernels[GEN_FLOATING], 1, sizeof(src_r_d), &src_r_d);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_FLOATING], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	clSetKernelArg(kernels[GEN_FLOATING], 0, sizeof(floating_g), &floating_g);
	clSetKernelArg(kernels[GEN_FLOATING], 1, sizeof(src_g_d), &src_g_d);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_FLOATING], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	clSetKernelArg(kernels[GEN_FLOATING], 0, sizeof(floating_b), &floating_b);
	clSetKernelArg(kernels[GEN_FLOATING], 1, sizeof(src_b_d), &src_b_d);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_FLOATING], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	// Gray
	clSetKernelArg(kernels[GEN_GRAY], 0, sizeof(gray), &gray);
	clSetKernelArg(kernels[GEN_GRAY], 1, sizeof(floating_r), &floating_r);
	clSetKernelArg(kernels[GEN_GRAY], 2, sizeof(floating_g), &floating_g);
	clSetKernelArg(kernels[GEN_GRAY], 3, sizeof(floating_b), &floating_b);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_GRAY], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	// gPyramid
	for (int k = 0; k < levels; k++) {
		// gPyramid[0]
		global_work_size[0] = width;
		global_work_size[1] = height;
		local_work_size[0] = 16;
		local_work_size[1] = 16;

		clSetKernelArg(kernels[GEN_GPYRAMID0], 0, sizeof(cl_mem), &gPyramid[0][k]);
		clSetKernelArg(kernels[GEN_GPYRAMID0], 1, sizeof(int), &k);
		clSetKernelArg(kernels[GEN_GPYRAMID0], 2, sizeof(cl_mem), &gray);
		err = clEnqueueNDRangeKernel(queue, kernels[GEN_GPYRAMID0], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		assert(err == CL_SUCCESS);

		for (int j = 1; j < maxJ; j++) {
			global_work_size[0] = (width >> j);
			global_work_size[1] = (height >> j);
			local_work_size[0] = (width >> j) > 16 ? 16 : (width >> j);
			local_work_size[1] = (height >> j) > 16 ? 16 : (height >> j);

			clSetKernelArg(kernels[DOWNSAMPLE_KERNEL], 0, sizeof(cl_mem), &gPyramid[j][k]);
			clSetKernelArg(kernels[DOWNSAMPLE_KERNEL], 1, sizeof(cl_mem), &gPyramid[j-1][k]);
			err = clEnqueueNDRangeKernel(queue, kernels[DOWNSAMPLE_KERNEL], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error: %d\n", err);
				return -1;
			}
		}
	}


	// inGPyramid[0]
	clReleaseMemObject(inGPyramid[0]);
	inGPyramid[0] = gray;
	clRetainMemObject(inGPyramid[0]);
	// inGPyramid
	for (int j = 1; j < maxJ; j++) {
		global_work_size[0] = (width >> j);
		global_work_size[1] = (height >> j);
		local_work_size[0] = (width >> j) > 16 ? 16 : (width >> j);
		local_work_size[1] = (height >> j) > 16 ? 16 : (height >> j);

		clSetKernelArg(kernels[DOWNSAMPLE_KERNEL], 0, sizeof(cl_mem), &inGPyramid[j]);
		clSetKernelArg(kernels[DOWNSAMPLE_KERNEL], 1, sizeof(cl_mem), &inGPyramid[j-1]);
		err = clEnqueueNDRangeKernel(queue, kernels[DOWNSAMPLE_KERNEL], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: %d\n", err);
			return -1;
		}
	}

	// outLPyramid
	for (int j = 0; j < maxJ - 1; j++) {
		global_work_size[0] = (width >> j);
		global_work_size[1] = (height >> j);
		local_work_size[0] = (width >> j) > 16 ? 16 : (width >> j);
		local_work_size[1] = (height >> j) > 16 ? 16 : (height >> j);

		clSetKernelArg(kernels[GEN_OUTLPYRAMID], 0, sizeof(cl_mem), &outLPyramid[j]);
		for (int arg = 0; arg < levels; arg++) {
			clSetKernelArg(kernels[GEN_OUTLPYRAMID], 1 + arg, sizeof(cl_mem), &gPyramid[j][arg]);
		}
		for (int arg = 0; arg < levels; arg++) {
			clSetKernelArg(kernels[GEN_OUTLPYRAMID], 1 + levels + arg, sizeof(cl_mem), &gPyramid[j+1][arg]);
		}
		clSetKernelArg(kernels[GEN_OUTLPYRAMID], 1 + 2 * levels, sizeof(cl_mem), &inGPyramid[j]);
		err = clEnqueueNDRangeKernel(queue, kernels[GEN_OUTLPYRAMID], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: %d\n", err);
			return -1;
		}
	}
	global_work_size[0] = (width >> (maxJ - 1));
	global_work_size[1] = (height >> (maxJ - 1));
	local_work_size[0] = (width >> (maxJ - 1)) > 16 ? 16 : (width >> (maxJ - 1));
	local_work_size[1] = (height >> (maxJ - 1)) > 16 ? 16 : (height >> (maxJ - 1));

	clSetKernelArg(kernels[GEN_OUTLPYRAMIDLOWEST], 0, sizeof(cl_mem), &outLPyramid[maxJ - 1]);
	for (int arg = 0; arg < levels; arg++) {
		clSetKernelArg(kernels[GEN_OUTLPYRAMIDLOWEST], 1 + arg, sizeof(cl_mem), &gPyramid[maxJ - 1][arg]);
	}
	clSetKernelArg(kernels[GEN_OUTLPYRAMIDLOWEST], 1 + levels, sizeof(cl_mem), &inGPyramid[maxJ - 1]);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_OUTLPYRAMIDLOWEST], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: %d\n", err);
		return -1;
	}

	
	// outGPyramid[maxJ - 1]
	err = clEnqueueCopyBuffer(queue, outLPyramid[maxJ - 1],
			outGPyramid[maxJ - 1], 0, 0, 
			sizeof(float) * (width >> (maxJ - 1)) * (height >> (maxJ - 1)), 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	// outGPyramid
	for (int j = maxJ - 2; j >= 0; j--) {
		global_work_size[0] = (width >> j);
		global_work_size[1] = (height >> j);
		local_work_size[0] = (width >> j) > 16 ? 16 : (width >> j);
		local_work_size[1] = (height >> j) > 16 ? 16 : (height >> j);

		clSetKernelArg(kernels[GEN_OUTGPYRAMID], 0, sizeof(cl_mem), &outGPyramid[j]);
		clSetKernelArg(kernels[GEN_OUTGPYRAMID], 1, sizeof(cl_mem), &outGPyramid[j+1]);
		clSetKernelArg(kernels[GEN_OUTGPYRAMID], 2, sizeof(cl_mem), &outLPyramid[j]);
		err = clEnqueueNDRangeKernel(queue, kernels[GEN_OUTGPYRAMID], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: %d\n", err);
			return -1;
		}
	}

	// output
	global_work_size[0] = width;
	global_work_size[1] = height;
	local_work_size[0] = 16;
	local_work_size[1] = 16;

	clSetKernelArg(kernels[GEN_OUTPUT], 0, sizeof(cl_mem), &dst_r_d);
	clSetKernelArg(kernels[GEN_OUTPUT], 1, sizeof(cl_mem), &outGPyramid[0]);
	clSetKernelArg(kernels[GEN_OUTPUT], 2, sizeof(cl_mem), &floating_r);
	clSetKernelArg(kernels[GEN_OUTPUT], 3, sizeof(cl_mem), &gPyramid[0][0]);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_OUTPUT], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	clSetKernelArg(kernels[GEN_OUTPUT], 0, sizeof(cl_mem), &dst_g_d);
	clSetKernelArg(kernels[GEN_OUTPUT], 1, sizeof(cl_mem), &outGPyramid[0]);
	clSetKernelArg(kernels[GEN_OUTPUT], 2, sizeof(cl_mem), &floating_g);
	clSetKernelArg(kernels[GEN_OUTPUT], 3, sizeof(cl_mem), &gPyramid[0][0]);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_OUTPUT], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	clSetKernelArg(kernels[GEN_OUTPUT], 0, sizeof(cl_mem), &dst_b_d);
	clSetKernelArg(kernels[GEN_OUTPUT], 1, sizeof(cl_mem), &outGPyramid[0]);
	clSetKernelArg(kernels[GEN_OUTPUT], 2, sizeof(cl_mem), &floating_b);
	clSetKernelArg(kernels[GEN_OUTPUT], 3, sizeof(cl_mem), &gPyramid[0][0]);
	err = clEnqueueNDRangeKernel(queue, kernels[GEN_OUTPUT], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	// Read buffer
	err = clEnqueueReadBuffer(queue, dst_r_d, CL_TRUE, 0, 
		sizeof(uint8_t) * width * height, dst_r, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	err = clEnqueueReadBuffer(queue, dst_g_d, CL_TRUE, 0, 
		sizeof(uint8_t) * width * height, dst_g, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	err = clEnqueueReadBuffer(queue, dst_b_d, CL_TRUE, 0, 
		sizeof(uint8_t) * width * height, dst_b, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	
	gettimeofday(&tim, NULL);  
    double dTime2 = tim.tv_sec+(tim.tv_usec/1000000.0); 

	returnRGB(dst_r, dst_g, dst_b);

	write_png_file(argv[2]);

	printf("Elapsed Time: %lf sec\n", dTime2 - dTime1);

	free(dst_b);
	free(dst_g);
	free(dst_r);

	free(src_b);
	free(src_g);
	free(src_r);

	// release cl buffers
	clReleaseMemObject(dst_b_d);
	clReleaseMemObject(dst_g_d);
	clReleaseMemObject(dst_r_d);
	for (int j = 0; j < maxJ; j++) {
		clReleaseMemObject(outGPyramid[j]);
	}
	for (int j = 0; j < maxJ; j++) {
		clReleaseMemObject(outLPyramid[j]);
	}
	for (int j = 0; j < maxJ; j++) {
		clReleaseMemObject(inGPyramid[j]);
	}	
	for (int j = 0; j < maxJ; j++) {
		for (int k = 0; k < levels; k++) {
			clReleaseMemObject(gPyramid[j][k]);
		}
	}
	clReleaseMemObject(gray);
	clReleaseMemObject(floating_b);
	clReleaseMemObject(floating_g);
	clReleaseMemObject(floating_r);
	clReleaseMemObject(src_b_d);
	clReleaseMemObject(src_g_d);
	clReleaseMemObject(src_r_d);
	clReleaseKernels(kernels);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

    return 0;
}

void read_png_file(char* file_name)
{
	unsigned char header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE *fp = fopen(file_name, "rb");
	if (!fp)
		abort_("[read_png_file] File %s could not be opened for reading", file_name);
	fread(header, 1, 8, fp);
	if (png_sig_cmp(header, 0, 8))
		abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
            abort_("[read_png_file] png_create_read_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
            abort_("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);


    /* read file */
    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during read_image");

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
            row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

    png_read_image(png_ptr, row_pointers);

    fclose(fp);
}


void write_png_file(char* file_name)
{
    /* create file */
    FILE *fp = fopen(file_name, "wb");
    if (!fp)
            abort_("[write_png_file] File %s could not be opened for writing", file_name);


    /* initialize stuff */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
            abort_("[write_png_file] png_create_write_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
            abort_("[write_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during init_io");

    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during writing header");

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during writing bytes");

    png_write_image(png_ptr, row_pointers);


    /* end write */
    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[write_png_file] Error during end of write");

    png_write_end(png_ptr, NULL);

    /* cleanup heap allocation */
    for (y=0; y<height; y++)
            free(row_pointers[y]);
    free(row_pointers);

    fclose(fp);
}


void process_file(void)
{
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB)
		abort_("[process_file] input file is PNG_COLOR_TYPE_RGB but must be PNG_COLOR_TYPE_RGBA "
				"(lacks the alpha channel)");

	if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA)
		abort_("[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (%d) (is %d)",
				PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));

        for (y=0; y<height; y++) {
                png_byte* row = row_pointers[y];
                for (x=0; x<width; x++) {
                        png_byte* ptr = &(row[x*4]);
                        //printf("Pixel at position [ %d - %d ] has RGBA values: %d - %d - %d - %d\n",
                        //       x, y, ptr[0], ptr[1], ptr[2], ptr[3]);

                        /* set red value to 0 and green value to the blue one */
                        ptr[0] = 0;
                        ptr[1] = ptr[2];
                }
        }
}

cl_program load_program(cl_context context, cl_device_id device, const char* filename)
{
	FILE *fp;
	size_t length;
	char *data;
	const char* source;
	size_t ret;

	// open file
	fp = fopen(filename, "rb");
	if(fp == NULL)
		perror("Error opening file\n");

	// get file length
	fseek (fp, 0, SEEK_END);
	length = ftell (fp);
	fseek (fp, 0, SEEK_SET);    // rewind(fp);

	// read program source
	data = (char*)malloc((length+1) * sizeof(char));
	ret = fread(data, sizeof(char), length, fp);
	if(ret != length)
		perror("Error reading file\n");
	data[length] = 0;

	// create and build program object
	source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
	if(program == 0) {
		perror("Error creating program\n");
		return 0;
	}

	// compile program
	if(clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
	{
		cl_int err;
		size_t len;
		char *buffer;

		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		buffer = calloc(sizeof(char), len);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		printf( "Error building program %d: %s\n", err, buffer);
		return 0;
	}

	free(data);
	fclose(fp);

	return program;
}

int clCreateKernels(cl_program program, cl_kernel **kernels_ptr)
{
	cl_int err;
	cl_kernel *kernels = (cl_kernel *)malloc(NUM_KERNELS * sizeof(cl_kernel));
	char kernels_name[NUM_KERNELS][35] = {
		"genFloating",
		"genGray",
		"genGPyramid0",
		"downSampleKernel",
		"genOutLPyramidLowest",
		"genOutLPyramid",
		"genOutGPyramid",
		"genOutput",
	};
	for (int i = 0; i < NUM_KERNELS; i++)
	{
		kernels[i] = clCreateKernel(program, kernels_name[i], &err);
		if (err != CL_SUCCESS)
		{
			printf("Create kernels error %d\n", err);
			return err;
		}
	}

	*kernels_ptr = kernels;

	return CL_SUCCESS;
}

int clReleaseKernels(cl_kernel *kernels)
{
	for (int i = 0; i < NUM_KERNELS; i++)
	{
		clReleaseKernel(kernels[i]);
	}

	return CL_SUCCESS;
}

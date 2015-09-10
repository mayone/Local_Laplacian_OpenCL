// File: local_laplacian.cl

#define levels 8
#define maxJ 8
#define alpha (1.0f / (levels - 1))
#define beta 1.0f

// Helper functions
float downSample(int x, int y, int width, int height, 
	__global float *src)
{
	float sum = 0.0f;
	
	sum += 1 * src[clamp(2 * y - 1, 0, height - 1) * width 
		+ clamp(2 * x - 1, 0, width - 1)];
	sum += 3 * src[clamp(2 * y - 1, 0, height - 1) * width 
		+ clamp(2 * x    , 0, width - 1)];
	sum += 3 * src[clamp(2 * y - 1, 0, height - 1) * width 
		+ clamp(2 * x + 1, 0, width - 1)];
	sum += 1 * src[clamp(2 * y - 1, 0, height - 1) * width 
		+ clamp(2 * x + 2, 0, width - 1)];
		
	sum += 3 * src[clamp(2 * y    , 0, height - 1) * width 
		+ clamp(2 * x - 1, 0, width - 1)];
	sum += 9 * src[clamp(2 * y    , 0, height - 1) * width 
		+ clamp(2 * x    , 0, width - 1)];
	sum += 9 * src[clamp(2 * y    , 0, height - 1) * width 
		+ clamp(2 * x + 1, 0, width - 1)];
	sum += 3 * src[clamp(2 * y    , 0, height - 1) * width 
		+ clamp(2 * x + 2, 0, width - 1)];
		
	sum += 3 * src[clamp(2 * y + 1, 0, height - 1) * width 
		+ clamp(2 * x - 1, 0, width - 1)];
	sum += 9 * src[clamp(2 * y + 1, 0, height - 1) * width 
		+ clamp(2 * x    , 0, width - 1)];
	sum += 9 * src[clamp(2 * y + 1, 0, height - 1) * width 
		+ clamp(2 * x + 1, 0, width - 1)];
	sum += 3 * src[clamp(2 * y + 1, 0, height - 1) * width 
		+ clamp(2 * x + 2, 0, width - 1)];
	
	sum += 1 * src[clamp(2 * y + 2, 0, height - 1) * width 
		+ clamp(2 * x - 1, 0, width - 1)];
	sum += 3 * src[clamp(2 * y + 2, 0, height - 1) * width 
		+ clamp(2 * x    , 0, width - 1)];
	sum += 3 * src[clamp(2 * y + 2, 0, height - 1) * width 
		+ clamp(2 * x + 1, 0, width - 1)];
	sum += 1 * src[clamp(2 * y + 2, 0, height - 1) * width 
		+ clamp(2 * x + 2, 0, width - 1)];
	
	return sum / 64.0f;
}

float upSample(int x, int y, int width, int height, 
	__global float *src)
{
	float sum = 0.0f;
	
	sum += 1 * src[clamp(y/2 - 1 + 2*(y%2), 0, height - 1) * width +
		clamp(x/2 - 1 + 2*(x%2), 0, width - 1)];
	sum += 3 * src[clamp(y/2 - 1 + 2*(y%2), 0, height - 1) * width +
		clamp(x/2, 0, width - 1)];
	sum += 3 * src[clamp(y/2, 0, height - 1) * width +
		clamp(x/2 - 1 + 2*(x%2), 0, width - 1)];
	sum += 9 * src[clamp(y/2, 0, height - 1) * width +
		clamp(x/2, 0, width - 1)];
	
	return sum / 16.0f;
}
/*
float remap(int input)
{
	float fx = (float)input / 256.0f;
	
	return alpha * fx * exp(-fx * fx / 2.0f);
}*/

// Kernel functions

// Convention:
// global_size - dest image size
// width, height - src image size
// x, y - dest pixel

// This function needs to be called 3 times for 3 channels.
__kernel
void genFloating(__global float *dest, __global uchar *src)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	
	dest[y * width + x] = (float)src[y * width + x] / 255.0f;
}

__kernel
void genGray(__global float *dest, __global float *r, 
	__global float *g, __global float *b)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	
	dest[y * width + x] = 
		0.299f * r[y * width + x] +
		0.587f * g[y * width + x] +
		0.114f * b[y * width + x];
}

__kernel
void genGPyramid0(__global float *dest, int k, 
	__global float *gray)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	
	float level = k * (1.0f / (levels - 1));
	float idx = gray[y * width + x] * (float)(levels - 1) * 256.0f;
	int idxi = clamp((int)idx,
		0, (levels - 1) * 256);
	float fx = (idxi - 256 * k) / 256.0f;
	float remap = alpha * fx * exp(-fx*fx/2.0f);
	dest[y * width + x] = gray[y * width + x] + 
		remap;
	//if (dest[y * width + x]  - gray[y * width + x] > 0.5f)
		//printf("%f->%f\n", gray[y * width + x], dest[y * width + x]);
}

__kernel
void downSampleKernel(__global float *dest, __global float *src)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	dest[y * width + x] = downSample(x, y, width * 2, height * 2, src);
}

/*
__kernel
void genLPyramid(__global float *dest, __global float *gPyramid, 
	__global float *gPyramidLow)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	dest[y * width + x] = 
		gPyramid[y * width + x] - 
		upSample(x, y, width/2, height/2, gPyramidLow);
}
*/

__kernel
void genOutLPyramid(__global float *dest,
	__global float *gPyramid0,
	__global float *gPyramid1,
	__global float *gPyramid2,
	__global float *gPyramid3,
	__global float *gPyramid4,
	__global float *gPyramid5,
	__global float *gPyramid6,
	__global float *gPyramid7,
	__global float *gPyramidLow0,
	__global float *gPyramidLow1,
	__global float *gPyramidLow2,
	__global float *gPyramidLow3,
	__global float *gPyramidLow4,
	__global float *gPyramidLow5,
	__global float *gPyramidLow6,
	__global float *gPyramidLow7,
	__global float *inGPyramid)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	__global float *gPyramid[levels];
	__global float *gPyramidLow[levels];
	
	gPyramid[0] = gPyramid0;
	gPyramid[1] = gPyramid1;
	gPyramid[2] = gPyramid2;
	gPyramid[3] = gPyramid3;
	gPyramid[4] = gPyramid4;
	gPyramid[5] = gPyramid5;
	gPyramid[6] = gPyramid6;
	gPyramid[7] = gPyramid7;
	gPyramidLow[0] = gPyramidLow0;
	gPyramidLow[1] = gPyramidLow1;
	gPyramidLow[2] = gPyramidLow2;
	gPyramidLow[3] = gPyramidLow3;
	gPyramidLow[4] = gPyramidLow4;
	gPyramidLow[5] = gPyramidLow5;
	gPyramidLow[6] = gPyramidLow6;
	gPyramidLow[7] = gPyramidLow7;
	
	float level = inGPyramid[y * width + x] * (levels - 1);
	int li = clamp((int)level, 0, levels - 2);
	float lf = level - (float)li;
	float lPyramid1 =
		gPyramid[li][y * width + x] - 
		upSample(x, y, width / 2, height / 2, gPyramidLow[li]);
	float lPyramid2 =
		gPyramid[li+1][y * width + x] - 
		upSample(x, y, width / 2, height / 2, gPyramidLow[li+1]);
	dest[y * width + x] = 
		(1.0f - lf) * lPyramid1 + lf * lPyramid2;
}
	
__kernel
void genOutLPyramidLowest(__global float *dest,
	__global float *gPyramid0,
	__global float *gPyramid1,
	__global float *gPyramid2,
	__global float *gPyramid3,
	__global float *gPyramid4,
	__global float *gPyramid5,
	__global float *gPyramid6,
	__global float *gPyramid7,
	__global float *inGPyramid)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	__global float *gPyramid[levels];
	
	gPyramid[0] = gPyramid0;
	gPyramid[1] = gPyramid1;
	gPyramid[2] = gPyramid2;
	gPyramid[3] = gPyramid3;
	gPyramid[4] = gPyramid4;
	gPyramid[5] = gPyramid5;
	gPyramid[6] = gPyramid6;
	gPyramid[7] = gPyramid7;
	
	float level = inGPyramid[y * width + x] * (levels - 1);
	int li = clamp((int)level, 0, levels - 2);
	float lf = level - (float)li;
	float lPyramid1 =
		gPyramid[li][y * width + x]; 
	float lPyramid2 =
		gPyramid[li+1][y * width + x]; 
	dest[y * width + x] = 
		(1.0f - lf) * lPyramid1 + lf * lPyramid2;
}
__kernel
void genOutGPyramid(__global float *dest, 
	__global float *outGPyramidLow, 
	__global float *outLPyramid)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	dest[y * width + x] = 
		upSample(x, y, width / 2, height / 2, outGPyramidLow) +
		outLPyramid[y * width + x];
}

// This function needs to be called 3 times for 3 channels.
// Please specify which channel of dest and floating to compute.
__kernel
void genOutput(__global uchar *dest, __global float *outGPyramid, 
	__global float *floating, __global float *gray)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	const float eps = 0.01f;

	float color = 
		outGPyramid[y * width + x] * 
		(floating[y * width + x] + eps) /
		(gray[y * width + x] + eps);
	dest[y * width + x] =
		(uchar)(clamp(color, 0.0f, 1.0f) * 255.0f);
}

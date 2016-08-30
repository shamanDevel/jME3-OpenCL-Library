
#define LSZ_2D(vi, sz)			((vi).x + (vi).y * (sz).x)
#define L_INVSZ_2D(i, vi, sz)	(vi).y = i / (sz).x; (vi).x = (i - (vi).y*(sz).x);
#define LSZ_3D(vi, sz)			((vi).x + ((vi).y + (vi).z * (sz).y) * (sz).x)
#define L_INVSZ_3D(i, vi, sz)	(vi).z = i / ((sz).x*(sz).y); (vi).y = (i - (vi).z*(sz).x*(sz).y)/(sz).x; (vi).x = i - (sz).x*((vi).y + (vi).z*(sz).y);

__kernel void Iteration2D(__global float* xIn, __global float* xOut, __global float* bIn, 
		__global float* AIn, int resX, int resY, __global float* residuum)
{
	const int idx = get_global_id(0);
	const int size = get_global_size(0);
	
	int2 ij;
	int2 res = (int2)(resX, resY);
	L_INVSZ_2D(idx, ij, res);

	float x00 = xIn[idx];
	float b = bIn[idx];
	float a00 = AIn[idx]; //A[i,j]
	float a10 = AIn[idx+1*size]; //A[i-1, j]
	float a20 = AIn[idx+2*size]; //A[i+1, j]
	float a01 = AIn[idx+3*size]; //A[i, j-1]
	float a02 = AIn[idx+4*size]; //A[i, j+1]
	
	float sum = a00*x00;
	if (ij.x>0) sum += xIn[LSZ_2D((int2)(ij.x-1, ij.y), res)] * a10;
	if (ij.x<resX-1) sum += xIn[LSZ_2D((int2)(ij.x+1, ij.y), res)] * a20;
	if (ij.y>0) sum += xIn[LSZ_2D((int2)(ij.x, ij.y-1), res)] * a01;
	if (ij.y<resY-1) sum += xIn[LSZ_2D((int2)(ij.x, ij.y+1), res)] * a02;
	float r = b - sum;
	residuum[idx] = r;
	xOut[idx] = x00 + r/a00;

	//printf("idx=%d, i=%d, j=%d, b=%2.3f, sum=%2.3f, a=(%2.2f, %2.2f, %2.2f, %2.2f, %2.2f), x=%2.2f\n", idx, ij.x, ij.y, b, sum, a00, a10, a20, a01, a02, x00);
}

__kernel void Iteration3D(__global float* xIn, __global float* xOut, __global float* bIn, 
		__global float* AIn, int resX, int resY, int resZ, __global float* residuum)
{
	const int idx = get_global_id(0);
	const int size = get_global_size(0);
	
	int3 ijk;
	int3 res = (int3)(resX, resY, resZ);
	L_INVSZ_3D(idx, ijk, res);

	float x000 = xIn[idx];
	float b = bIn[idx];
	float a000 = AIn[idx]; //A[i,j]
	float a100 = AIn[idx+1*size]; //A[i-1, j, k]
	float a200 = AIn[idx+2*size]; //A[i+1, j, k]
	float a010 = AIn[idx+3*size]; //A[i, j-1, k]
	float a020 = AIn[idx+4*size]; //A[i, j+1, k]
	float a001 = AIn[idx+5*size]; //A[i, j, k-1]
	float a002 = AIn[idx+6*size]; //A[i, j, k+1]
	
	float sum = a000*x000;
	if (ijk.x>0) sum += xIn[LSZ_3D((int3)(ijk.x-1, ijk.y, ijk.z), res)] * a100;
	if (ijk.x<resX-1) sum += xIn[LSZ_3D((int3)(ijk.x+1, ijk.y, ijk.z), res)] * a200;
	if (ijk.y>0) sum += xIn[LSZ_3D((int3)(ijk.x, ijk.y-1, ijk.z), res)] * a010;
	if (ijk.y<resY-1) sum += xIn[LSZ_3D((int3)(ijk.x, ijk.y+1, ijk.z), res)] * a020;
	if (ijk.z>0) sum += xIn[LSZ_3D((int3)(ijk.x, ijk.y, ijk.z-1), res)] * a001;
	if (ijk.z<resZ-1) sum += xIn[LSZ_3D((int3)(ijk.x, ijk.y, ijk.z+1), res)] * a002;
	float r = b - sum;
	residuum[idx] = r;
	xOut[idx] = x000 + r/a000;
}

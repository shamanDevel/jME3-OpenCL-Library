
//Defined by the host code
//#define TYPE float (for example)

#define SIZE_T long

__kernel void Fill(__global TYPE* b, TYPE val, SIZE_T offset, SIZE_T step)
{
	unsigned int id = get_global_id(0);
	b[offset + step*id] = val;
}

__kernel void AXPY(TYPE a, __global TYPE* x, __global TYPE* y, __global TYPE* dest,
		SIZE_T offsetX, SIZE_T offsetY, SIZE_T offsetDest,
		SIZE_T stepX, SIZE_T stepY, SIZE_T stepDest)
{
	unsigned int id = get_global_id(0);
	dest[offsetDest + stepDest * id] = a * x[offsetX + stepX * id] + y[offsetY + stepY * id];
}


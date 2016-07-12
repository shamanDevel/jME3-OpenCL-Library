
//Defined by the host code
//#define TYPE float (for example)
//#define IS_FLOAT_TYPE 1/0

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
#if IS_FLOAT_TYPE==1
	dest[offsetDest + stepDest * id] = fma(a, x[offsetX + stepX * id], y[offsetY + stepY * id]);
#else
	dest[offsetDest + stepDest * id] = a * x[offsetX + stepX * id] + y[offsetY + stepY * id];
#endif
}

#define MAP_TEMPLATE(op, name) \
	__kernel void MAP_##name (__global TYPE* x, TYPE arg, __global TYPE* dest, \
			SIZE_T offsetX, SIZE_T offsetDest, SIZE_T stepX, SIZE_T stepDest){ \
		unsigned int id = get_global_id(0); \
		TYPE a = x[offsetX + stepX * id]; \
		TYPE b = arg; \
		TYPE c = op ; \
		dest[offsetDest + stepDest * id] = c; \
	}

MAP_TEMPLATE(b, SET)
MAP_TEMPLATE(a+b, ADD)
MAP_TEMPLATE(a-b, SUB)
MAP_TEMPLATE(b-a, SUB_INV)
MAP_TEMPLATE(a*b, MUL)
MAP_TEMPLATE(a/b, DIV)
MAP_TEMPLATE(b/a, DIV_INV)
//The following specializations still throw a compiler error: ambigious functions
#if IS_FLOAT_TYPE==1
MAP_TEMPLATE(fabs(a), ABS)
MAP_TEMPLATE(exp(a), EXP)
MAP_TEMPLATE(log(a)/log(b), LOG)
MAP_TEMPLATE(pow(a, b), POW)
MAP_TEMPLATE(pow(b, a), POW_INV)
#else
MAP_TEMPLATE(abs(a), ABS)
MAP_TEMPLATE((TYPE) exp((double)a), EXP)
MAP_TEMPLATE((TYPE) (log((double)a)/log((double)b)), LOG)
MAP_TEMPLATE((TYPE) pow((double)a, (double)b), POW)
MAP_TEMPLATE((TYPE) pow((double)b, (double)a), POW_INV)
#endif


//Defined by the host code
//#define TYPE float (for example)
//#define IS_FLOAT_TYPE 1/0
//#define TYPE_MIN
//#define TYPE_MAX

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
	__kernel void Map_##name (__global TYPE* x, TYPE arg, __global TYPE* dest, \
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
MAP_TEMPLATE((TYPE) exp((float)a), EXP)
MAP_TEMPLATE((TYPE) (log((float)a)/log((float)b)), LOG)
MAP_TEMPLATE((TYPE) pow((float)a, (float)b), POW)
MAP_TEMPLATE((TYPE) pow((float)b, (float)a), POW_INV)
#endif

#define REDUCE_TEMPLATE(op1, op2, neutralElement, name) \
	__kernel void Reduce_##name (__global TYPE* buffer, __local TYPE* scratch, __const int length, __global TYPE* result, __const int offset, __const int step) \
	{ \
		int global_index = get_global_id(0); \
		TYPE accumulator = neutralElement; \
		while (global_index < length) \
		{ \
			TYPE a = buffer[offset + step * global_index]; \
			TYPE b = accumulator; \
			accumulator = op1; \
			global_index += get_global_size(0); \
		} \
		int local_index = get_local_id(0); \
		scratch[local_index] = accumulator; \
		barrier(CLK_LOCAL_MEM_FENCE); \
		for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) \
		{ \
			if (local_index < offset) \
			{ \
				TYPE a = scratch[local_index + offset]; \
				TYPE b = scratch[local_index]; \
				scratch[local_index] = op2; \
			} \
			barrier(CLK_LOCAL_MEM_FENCE); \
		} \
		if (local_index == 0) \
		{ \
			result[get_group_id(0)] = scratch[0]; \
		} \
	}


REDUCE_TEMPLATE(a+b , a+b , 0, NONE_ADD)
REDUCE_TEMPLATE(a*b , a*b , 1, NONE_MUL)
#if IS_FLOAT_TYPE==1
REDUCE_TEMPLATE(fmin(a,b) , fmin(a,b) , TYPE_MAX, NONE_MIN)
REDUCE_TEMPLATE(fmax(a,b) , fmax(a,b) , TYPE_MIN, NONE_MAX)
#else
REDUCE_TEMPLATE(min(a,b) , min(a,b) , TYPE_MAX, NONE_MIN)
REDUCE_TEMPLATE(max(a,b) , max(a,b) , TYPE_MIN, NONE_MAX)
#endif

#if IS_FLOAT_TYPE==1
REDUCE_TEMPLATE(fabs(a)+b , a+b , 0, ABS_ADD)
REDUCE_TEMPLATE(fabs(a)*b , a*b , 1, ABS_MUL)
REDUCE_TEMPLATE(fmin(fabs(a),b) , fmin(a,b) , TYPE_MAX, ABS_MIN)
REDUCE_TEMPLATE(fmax(fabs(a),b) , fmax(a,b) , TYPE_MIN, ABS_MAX)
#else
REDUCE_TEMPLATE(abs(a)+b , a+b , 0, ABS_ADD)
REDUCE_TEMPLATE(abs(a)*b , a*b , 0, ABS_MUL)
REDUCE_TEMPLATE(min((TYPE) abs(a),b) , min(a,b) , TYPE_MAX, ABS_MIN)
REDUCE_TEMPLATE(max((TYPE) abs(a),b) , max(a,b) , TYPE_MIN, ABS_MAX)
#endif

REDUCE_TEMPLATE(a*a+b , a+b , 0, SQUARE_ADD)
REDUCE_TEMPLATE(a*a*b , a*b , 1, SQUARE_MUL)
#if IS_FLOAT_TYPE==1
REDUCE_TEMPLATE(fmin(a*a,b) , fmin(a,b) , TYPE_MAX, SQUARE_MIN)
REDUCE_TEMPLATE(fmax(a*a,b) , fmax(a,b) , TYPE_MIN, SQUARE_MAX)
#else
REDUCE_TEMPLATE(min(a*a,b) , min(a,b) , TYPE_MAX, SQUARE_MIN)
REDUCE_TEMPLATE(max(a*a,b) , max(a,b) , TYPE_MIN, SQUARE_MAX)
#endif

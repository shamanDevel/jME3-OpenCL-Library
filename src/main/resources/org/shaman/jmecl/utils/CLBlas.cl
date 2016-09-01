
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

#define CONCAT(a, b) a##b

#define REORDER_TEMPLATE(type, name) \
	__kernel void Reorder_##name (__global uint* indices, __global type *src, __global type *dst) \
	{ \
		const int idx = get_global_id(0); \
		dst[idx] = src[indices[idx]]; \
	}
REORDER_TEMPLATE(TYPE, 1)
REORDER_TEMPLATE(CONCAT(TYPE,2), 2)
REORDER_TEMPLATE(CONCAT(TYPE,3), 3)
REORDER_TEMPLATE(CONCAT(TYPE,4), 4)

__kernel void FillIndices(__global TYPE* x, TYPE start, TYPE step)
{
	unsigned int id = get_global_id(0);
	x[id] = start + id * step;
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
REDUCE_TEMPLATE(min((TYPE) (a*a),b) , min(a,b) , TYPE_MAX, SQUARE_MIN)
REDUCE_TEMPLATE(max((TYPE) (a*a),b) , max(a,b) , TYPE_MIN, SQUARE_MAX)
#endif

#define REDUCE2_TEMPLATE(op1, op2, neutralElement, name) \
	__kernel void Reduce2_##name (__global TYPE* buffer1, __global TYPE* buffer2, __local TYPE* scratch, __const int length, __global TYPE* result, __const int offset1, __const int step1, __const int offset2, __const int step2) \
	{ \
		int global_index = get_global_id(0); \
		TYPE accumulator = neutralElement; \
		while (global_index < length) \
		{ \
			TYPE a = buffer1[offset1 + step1 * global_index]; \
			TYPE b = buffer2[offset2 + step2 * global_index]; \
			TYPE elem = op1; \
			a = elem; \
			b = accumulator; \
			accumulator = op2; \
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

REDUCE2_TEMPLATE(a+b , a+b , 0, ADD_ADD)
REDUCE2_TEMPLATE(a+b , a*b , 1, ADD_MUL)
#if IS_FLOAT_TYPE==1
REDUCE2_TEMPLATE(a+b , fmin(a,b) , TYPE_MAX, ADD_MIN)
REDUCE2_TEMPLATE(a+b , fmax(a,b) , TYPE_MIN, ADD_MAX)
#else
REDUCE2_TEMPLATE(a+b , min(a,b) , TYPE_MAX, ADD_MIN)
REDUCE2_TEMPLATE(a+b , max(a,b) , TYPE_MIN, ADD_MAX)
#endif

REDUCE2_TEMPLATE(a-b , a+b , 0, SUB_ADD)
REDUCE2_TEMPLATE(a-b , a*b , 1, SUB_MUL)
#if IS_FLOAT_TYPE==1
REDUCE2_TEMPLATE(a-b , fmin(a,b) , TYPE_MAX, SUB_MIN)
REDUCE2_TEMPLATE(a-b , fmax(a,b) , TYPE_MIN, SUB_MAX)
#else
REDUCE2_TEMPLATE(a-b , min(a,b) , TYPE_MAX, SUB_MIN)
REDUCE2_TEMPLATE(a-b , max(a,b) , TYPE_MIN, SUB_MAX)
#endif

REDUCE2_TEMPLATE(a*b , a+b , 0, MUL_ADD)
REDUCE2_TEMPLATE(a*b , a*b , 1, MUL_MUL)
#if IS_FLOAT_TYPE==1
REDUCE2_TEMPLATE(a*b , fmin(a,b) , TYPE_MAX, MUL_MIN)
REDUCE2_TEMPLATE(a*b , fmax(a,b) , TYPE_MIN, MUL_MAX)
#else
REDUCE2_TEMPLATE(a*b , min(a,b) , TYPE_MAX, MUL_MIN)
REDUCE2_TEMPLATE(a*b , max(a,b) , TYPE_MIN, MUL_MAX)
#endif

#if IS_FLOAT_TYPE==1
REDUCE2_TEMPLATE(fmin(a,b) , a+b , 0, MIN_ADD)
REDUCE2_TEMPLATE(fmin(a,b) , a*b , 1, MIN_MUL)
REDUCE2_TEMPLATE(fmin(a,b) , fmin(a,b) , TYPE_MAX, MIN_MIN)
REDUCE2_TEMPLATE(fmin(a,b) , fmax(a,b) , TYPE_MIN, MIN_MAX)
#else
REDUCE2_TEMPLATE(min(a,b) , a+b , 0, MIN_ADD)
REDUCE2_TEMPLATE(min(a,b) , a*b , 1, MIN_MUL)
REDUCE2_TEMPLATE(min(a,b) , min(a,b) , TYPE_MAX, MIN_MIN)
REDUCE2_TEMPLATE(min(a,b) , max(a,b) , TYPE_MIN, MIN_MAX)
#endif

#if IS_FLOAT_TYPE==1
REDUCE2_TEMPLATE(fmax(a,b) , a+b , 0, MAX_ADD)
REDUCE2_TEMPLATE(fmax(a,b) , a*b , 1, MAX_MUL)
REDUCE2_TEMPLATE(fmax(a,b) , fmin(a,b) , TYPE_MAX, MAX_MIN)
REDUCE2_TEMPLATE(fmin(a,b) , fmax(a,b) , TYPE_MIN, MAX_MAX)
#else
REDUCE2_TEMPLATE(max(a,b) , a+b , 0, MAX_ADD)
REDUCE2_TEMPLATE(max(a,b) , a*b , 1, MAX_MUL)
REDUCE2_TEMPLATE(max(a,b) , min(a,b) , TYPE_MAX, MAX_MIN)
REDUCE2_TEMPLATE(max(a,b) , max(a,b) , TYPE_MIN, MAX_MAX)
#endif

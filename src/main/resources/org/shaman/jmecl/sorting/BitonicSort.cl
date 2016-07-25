
//Input: KEY_TYPE, VALUE_TYPE, COMPARISON_GREATER(x, y)

typedef KEY_TYPE key_t;
typedef VALUE_TYPE value_t;

inline bool greater(key_t a, key_t b) {  //returns a<=b
	return COMPARISON_GREATER(a, b);
}

inline void compareAndSwap(__global key_t* keys, __global value_t* values, int i, int j)
{
	key_t k1 = keys[i];
	key_t k2 = keys[j];
	if (!greater(k1, k2)) {
		keys[j] = k1;
		keys[i] = k2;
		value_t tmp = values[i];
		values[i] = values[j];
		values[j] = tmp;
	}
}

inline void compareAndSwapS(__local key_t* keys, __local value_t* values, int i, int j)
{
	key_t k1 = keys[i];
	key_t k2 = keys[j];
	if (!greater(k1, k2)) {
		keys[j] = k1;
		keys[i] = k2;
		value_t tmp = values[i];
		values[i] = values[j];
		values[j] = tmp;
	}
}

__kernel void BitonicTrivial(__global key_t* keys, __global value_t* values, int phase, int step)
{
	const int idx = get_global_id(0);
	//unnormalized bitonic network
	int stepSize = 1 << (step-1);
	int part = idx / stepSize;
	int start = (part * 2 * stepSize) + (idx % stepSize);
	int direction = (idx / (1 << (phase-1))) % 2;
	int i,j;
	if (direction == 0) {
		i = start;
		j = start + stepSize;
	} else {
		j = start;
		i = start + stepSize;
	}
	compareAndSwap(keys, values, i, j);
}

__kernel void BitonicShared(__global key_t* keys, __global value_t* values, int phase, int step_, __local key_t* sharedKeys, __local value_t* sharedValues)
{
	const int idx = get_global_id(0);
	const int idxl = get_local_id(0);
	const int local_size = get_local_size(0);
	int offset = get_group_id(0) * local_size * 2;

	//1. load it into shared memory
	sharedKeys[idxl] = keys[idxl + offset];
	sharedValues[idxl] = values[idxl + offset];
	sharedKeys[idxl + local_size] = keys[idxl + local_size + offset];
	sharedValues[idxl + local_size] = values[idxl + local_size + offset];

	barrier(CLK_LOCAL_MEM_FENCE);

	//2. sort locally
	for (int step = step_; step>=1; --step) {
		int stepSize = 1 << (step-1);
		int part = idx / stepSize;
		int start = (part * 2 * stepSize) + (idx % stepSize);
		int direction = (idx / (1 << (phase-1))) % 2;
		int i,j;
		if (direction == 0) {
			i = start;
			j = start + stepSize;
		} else {
			j = start;
			i = start + stepSize;
		}
		compareAndSwapS(sharedKeys, sharedValues, i - offset, j - offset);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//3. write back
	keys[idxl + offset] = sharedKeys[idxl];
	values[idxl + offset] = sharedValues[idxl];
	keys[idxl + local_size + offset] = sharedKeys[idxl + local_size];
	values[idxl + local_size + offset] = sharedValues[idxl + local_size];
}
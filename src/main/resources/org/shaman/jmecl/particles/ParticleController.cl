
__kernel void FindFirstOne(__global int* sortedDeletionBuffer, __global int* output)
{
	const int idx = get_global_id(0);
	if (sortedDeletionBuffer[idx]==1) {
		if (idx == 0 || (idx>0 && sortedDeletionBuffer[idx-1]==0)) {
			output[0] = idx;
		}
	}
}

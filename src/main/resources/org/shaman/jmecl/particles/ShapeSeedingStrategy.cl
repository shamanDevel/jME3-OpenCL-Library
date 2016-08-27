#import "Common/OpenCL/Random.clh"

__kernel void SeedPoint(int offset, __global float4* positions, float4 pointPos, float tpf, __global ulong* seeds)
{
	const int idx = get_global_id(0);
	float time = randFloat(seeds + idx) * tpf;
	positions[idx + offset] = (float4)(pointPos.xyz, -time);
}

__kernel void SeedSphere(int offset, __global float4* positions, float4 center, float radius, float tpf, __global ulong* seeds)
{
	const int idx = get_global_id(0);
	float time = randFloat(seeds + idx) * tpf;
	float3 pos;
	do {
		pos.x = (randFloat(seeds + idx) * 2 - 1) * radius;
		pos.y = (randFloat(seeds + idx) * 2 - 1) * radius;
		pos.z = (randFloat(seeds + idx) * 2 - 1) * radius;
	} while (length(pos) > radius);
	pos += center.xyz;
	positions[idx + offset] = (float4)(pos, -time);
}

__kernel void SeedBox(int offset, __global float4* positions, float4 minBox, float4 maxBox, float tpf, __global ulong* seeds)
{
	const int idx = get_global_id(0);
	float time = randFloat(seeds + idx) * tpf;
	float3 pos = (float3)(
		minBox.x + (maxBox.x-minBox.x)*randFloat(seeds + idx),
		minBox.y + (maxBox.y-minBox.y)*randFloat(seeds + idx),
		minBox.z + (maxBox.z-minBox.z)*randFloat(seeds + idx)
	);
	positions[idx + offset] = (float4)(pos, -time);
}

__kernel void InitParticles(int offset, __global float4* velocities, 
		float4 initialVelocity, float velocityVariation, float initialDensity, float densityVariation, __global ulong* seeds)
{
	const int idx = get_global_id(0);
	float2 gauss1 = randGaussianf(seeds + idx);
	float2 gauss2 = randGaussianf(seeds + idx);
	float3 vel = initialVelocity.xyz + velocityVariation * (float3)(gauss1, gauss2.x);
	float density = fmax((float) 0, initialDensity + densityVariation * gauss2.y);
	velocities[idx + offset] = (float4)(vel, density);
}
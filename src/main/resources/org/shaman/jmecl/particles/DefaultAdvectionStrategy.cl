

__kernel void DeletionBox(__global float4* positions, __global int* deletionBuffer,
		float4 minBox, float4 maxBox)
{
	const int idx = get_global_id(0);
	float3 position = positions[idx].xyz;
	if (any(isless(position, minBox.xyz)) || any(isgreater(position, maxBox.xyz)))
		deletionBuffer[idx] = 1;
}

__kernel void DeletionSphere(__global float4* positions, __global int* deletionBuffer, float4 centerAndRadius)
{
	const int idx = get_global_id(0);
	float3 position = positions[idx].xyz;
	float3 center = centerAndRadius.xyz;
	float radiusSqr = centerAndRadius.w * centerAndRadius.w;
	float3 p = position - center;
	if (dot(p, p) > radiusSqr)
		deletionBuffer[idx] = 1;
}

__kernel void DeletionDensityThreshold(__global float4* positions, __global int* deletionBuffer, float threshold)
{
	const int idx = get_global_id(0);
	if (positions[idx].w < threshold)
		deletionBuffer[idx] = 1;
}

__kernel void Advect(__global float4* positions, __global float4* velocities, __global float* temperatures,
		float alpha, float beta, float4 gravity,
		float lambda, float mu, float dt)
{
	const int idx = get_global_id(0);
	float3 position = positions[idx].xyz;
	float density = positions[idx].w;
	float3 velocity = velocities[idx].xyz;
	float time = velocities[idx].w;
	float temperature = temperatures[idx];
	
	if (time < 0) dt -= time;
	time += dt;

	velocity += dt * (alpha * density * gravity.xyz - beta * temperature * gravity.xyz);
	position += dt * velocity;

	temperature -= dt * lambda * temperature;
	density -= dt * mu * density;

	positions[idx] = (float4)(position, density);
	velocities[idx] = (float4)(velocity, time);
	temperatures[idx] = temperature;
}
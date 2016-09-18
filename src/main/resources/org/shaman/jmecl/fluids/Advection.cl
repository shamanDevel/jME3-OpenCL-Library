
#import "org/shaman/jmecl/fluids/Grids.clh"

__kernel void SemiLagrangeReal2D(RealGrid_t src, RealGrid_t dst, MACGrid_t vel, float dt, int sizeX, int sizeY)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	float3 v = MACGrid_getCentered2D(vel, pos, dim);
	GridPosition_t posf = (GridPosition_t)(pos.x+0.5, pos.y+0.5, 0) - v * dt;
	posf = Grid_clampPos(posf, dim);
	dst[idx] = RealGrid_getInterpolated(src, posf, dim);
	//printf("p=%3.2d,%3.2d,%3.2d : v=%3.2f,%3.2f,%3.2f : density=%3.2f->%3.2f\n", pos.x, pos.y, pos.z, v.x, v.y, v.z, src[idx], dst[idx]);
	//printf("p=%3.2d,%3.2d,%3.2d : v=%3.2f,%3.2f,%3.2f\n", pos.x, pos.y, pos.z, v.x, v.y, v.z);
}

__kernel void SemiLagrangeMAC2D(MACGrid_t src, MACGrid_t dst, MACGrid_t vel, float dt, int sizeX, int sizeY)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	if (pos.x==0 || pos.y==0) return;
	float3 xpos = (float3)(pos.x+0.5, pos.y+0.5, pos.z+0.5) - MACGrid_getAtMACX2D(vel, pos, dim);
	float vx = MACGrid_getInterpolatedX(src, xpos, dim);
	float3 ypos = (float3)(pos.x+0.5, pos.y+0.5, pos.z+0.5) - MACGrid_getAtMACY2D(vel, pos, dim);
	float vy = MACGrid_getInterpolatedY(src, ypos, dim);

	*MACGrid_getPointerX(dst, pos, dim) = vx;
	*MACGrid_getPointerY(dst, pos, dim) = vy;
}
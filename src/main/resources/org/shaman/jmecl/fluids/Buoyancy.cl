
#import "org/shaman/jmecl/fluids/Grids.clh"

__kernel void AddSimpleBuoyancy2D(FlagGrid_t flags, RealGrid_t factor, MACGrid_t vel, float2 strength, int sizeX, int sizeY)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	//if (!Grid_isValid(pos, dim)) return;
	if (FlagGrid_get(flags, pos, dim) != CellType_Fluid) return;
	if (pos.x>0 && FlagGrid_get(flags, pos-(int3)(1,0,0), dim) == CellType_Fluid) {
		*MACGrid_getPointerX(vel, pos, dim) += (0.5 * strength.x) * (RealGrid_get(factor, pos, dim) + RealGrid_get(factor, pos-(int3)(1,0,0), dim));
	}
	if (pos.y>0 && FlagGrid_get(flags, pos-(int3)(0,1,0), dim) == CellType_Fluid) {
		*MACGrid_getPointerY(vel, pos, dim) += (0.5 * strength.y) * (RealGrid_get(factor, pos, dim) + RealGrid_get(factor, pos-(int3)(0,1,0), dim));
	}
}


__kernel void AddSimpleBuoyancy3D(FlagGrid_t flags, RealGrid_t factor, MACGrid_t vel, float4 strength, int sizeX, int sizeY, int sizeZ)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, sizeZ);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	//if (!Grid_isValid(pos, dim)) return;
	if (FlagGrid_get(flags, pos, dim) != CellType_Fluid) return;
	if (pos.x>0 && FlagGrid_get(flags, pos-(int3)(1,0,0), dim) == CellType_Fluid) {
		*MACGrid_getPointerX(vel, pos, dim) += (0.5 * strength.x) * (RealGrid_get(factor, pos, dim) + RealGrid_get(factor, pos-(int3)(1,0,0), dim));
	}
	if (pos.y>0 && FlagGrid_get(flags, pos-(int3)(0,1,0), dim) == CellType_Fluid) {
		*MACGrid_getPointerY(vel, pos, dim) += (0.5 * strength.y) * (RealGrid_get(factor, pos, dim) + RealGrid_get(factor, pos-(int3)(0,1,0), dim));
	}
	if (pos.z>0 && FlagGrid_get(flags, pos-(int3)(0,0,1), dim) == CellType_Fluid) {
		*MACGrid_getPointerZ(vel, pos, dim) += (0.5 * strength.z) * (RealGrid_get(factor, pos, dim) + RealGrid_get(factor, pos-(int3)(0,0,1), dim));
	}
}

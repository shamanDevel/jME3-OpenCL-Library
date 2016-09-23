
#import "org/shaman/jmecl/fluids/Grids.clh"

__kernel void SetFlagsRect2D(FlagGrid_t flags, int flag, int offsetX, int offsetY, int sizeX, int sizeY, int resX, int resY)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	pos += (int3)(offsetX, offsetY, 0);
	dim = (int3)(resX, resY, 1);
	if (!Grid_isValid(pos, dim)) return;
	FlagGrid_set(flags, pos, dim, flag);
}

__kernel void ApplyDirichlet2D(RealGrid_t target, FlagGrid_t flags, int flag, float value, int sizeX, int sizeY)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	if (FlagGrid_get(flags, pos, dim) & flag)
		RealGrid_set(target, pos, dim, value);
}
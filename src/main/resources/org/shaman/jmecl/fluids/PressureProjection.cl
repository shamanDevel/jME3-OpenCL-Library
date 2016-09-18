
#import "org/shaman/jmecl/fluids/Grids.clh"

__kernel void MakeRhs2D(FlagGrid_t flags, RealGrid_t rhs, MACGrid_t vel, int sizeX, int sizeY)
{
	int idx = get_global_id(0);

	if ((flags[idx] & CellType_Fluid) == 0) {
		rhs[idx] = 0;
		return;
	}

	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);

	float set = 0;
	set += MACGrid_getX(vel, pos, dim) - MACGrid_getX(vel, pos+(int3)(1,0,0), dim);
	set += MACGrid_getY(vel, pos, dim) - MACGrid_getY(vel, pos+(int3)(0,1,0), dim);
	rhs[idx] = set;
}

__kernel void MakeLaplaceMatrix2D(FlagGrid_t flags, RealGrid_t A0, RealGrid_t Ai, RealGrid_t Aj, int sizeX, int sizeY)
{
	int idx = get_global_id(0);
	if ((flags[idx] & CellType_Fluid) == 0) {
		return;
	}

	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);

	//diagonals
	float a0 = 0;
	if (pos.x>0 && !(FlagGrid_get(flags, pos-(int3)(1,0,0), dim) & CellType_Obstacle)) a0 += 1;
	if (pos.x<sizeX-1 && !(FlagGrid_get(flags, pos+(int3)(1,0,0), dim) & CellType_Obstacle)) a0 += 1;
	if (pos.y>0 && !(FlagGrid_get(flags, pos-(int3)(0,1,0), dim) & CellType_Obstacle)) a0 += 1;
	if (pos.y<sizeY-1 && !(FlagGrid_get(flags, pos+(int3)(0,1,0), dim) & CellType_Obstacle)) a0 += 1;
	A0[idx] = a0;

	//off-diagonals
	if (pos.x>0 && (FlagGrid_get(flags, pos-(int3)(1,0,0), dim) & CellType_Fluid)) Ai[idx] = -1;
	if (pos.y>0 && (FlagGrid_get(flags, pos-(int3)(0,1,0), dim) & CellType_Fluid)) Aj[idx] = -1;
}

__kernel void CorrectVelocity2D(FlagGrid_t flags, MACGrid_t vel, RealGrid_t pressure, int sizeX, int sizeY)
{
	int idx = get_global_id(0);
	GridSize_t dim = (int3)(sizeX, sizeY, 1);
	GridIndex_t pos;
	L_INVSZ(idx, pos, dim);
	if (flags[idx] & CellType_Fluid) 
	{
		if (pos.x>0 && (FlagGrid_get(flags, pos-(int3)(1,0,0), dim) & CellType_Fluid)) 
			*MACGrid_getPointerX(vel, pos, dim) -= (pressure[idx] - RealGrid_get(pressure, pos-(int3)(1,0,0), dim));
		else if (pos.x>0 && (FlagGrid_get(flags, pos-(int3)(1,0,0), dim) & CellType_Empty))
			*MACGrid_getPointerX(vel, pos, dim) -= pressure[idx];
		
		if (pos.y>0 && (FlagGrid_get(flags, pos-(int3)(0,1,0), dim) & CellType_Fluid)) 
			*MACGrid_getPointerY(vel, pos, dim) -= (pressure[idx] - RealGrid_get(pressure, pos-(int3)(0,1,0), dim));
		else if (pos.y>0 && (FlagGrid_get(flags, pos-(int3)(0,1,0), dim) & CellType_Empty))
			*MACGrid_getPointerY(vel, pos, dim) -= pressure[idx];
	}
	else if ((flags[idx] & CellType_Empty) && !(flags[idx] & CellType_Outflow))
	{
		if (pos.x>0 && (FlagGrid_get(flags, pos-(int3)(1,0,0), dim) & CellType_Fluid))
			*MACGrid_getPointerX(vel, pos, dim) += RealGrid_get(pressure, pos-(int3)(1,0,0), dim);
		else
			*MACGrid_getPointerX(vel, pos, dim) = 0;

		if (pos.y>0 && (FlagGrid_get(flags, pos-(int3)(0,1,0), dim) & CellType_Fluid))
			*MACGrid_getPointerY(vel, pos, dim) += RealGrid_get(pressure, pos-(int3)(0,1,0), dim);
		else
			*MACGrid_getPointerY(vel, pos, dim) = 0;
	}
}
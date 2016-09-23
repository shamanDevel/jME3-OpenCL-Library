/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.opencl.Kernel;
import com.jme3.opencl.Program;

/**
 *
 * @author Sebastian
 */
public class BoundaryTools {
	private static final String SOURCE_FILE = "org/shaman/jmecl/fluids/BoundaryTools.cl";
	
	private final FluidSolver solver;
	
	private final Kernel setFlagsRect2DKernel;
	private final Kernel applyDirichlet2DKernel;

	public BoundaryTools(FluidSolver solver) {
		this.solver = solver;
		
		String cacheID = BoundaryTools.class.getName();
		Program program = solver.clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			program = solver.clSettings.getClContext().createProgramFromSourceFiles(solver.clSettings.getAssetManager(), SOURCE_FILE);
			program.build();
			solver.clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		setFlagsRect2DKernel = program.createKernel("SetFlagsRect2D");
		applyDirichlet2DKernel = program.createKernel("ApplyDirichlet2D");
	}
	
	/**
	 * Sets the flags in the specified box area to the specified flag
	 * @param flags the flag grid
	 * @param type the cell type to set
	 * @param origin the origin of the box, 2 coordinates if 2D, 3 if 3D mode
	 * @param size the size of the box, 2 coordinates if 2D, 3 if 3D mode
	 */
	public void setFlagsInRect(FlagGrid flags, int type, int[] origin, int[] size)
	{
		if (solver.is2D()) {
			if (origin.length != 2 || size.length != 2) {
				throw new IllegalArgumentException("arrays must have length 2");
			}
			Kernel.WorkSize ws = new Kernel.WorkSize(size[0] * size[1]);
			setFlagsRect2DKernel.Run1NoEvent(solver.clSettings.getClCommandQueue(), ws, 
					flags.getBuffer(), type, origin[0], origin[1], size[0], size[1], 
					solver.getResolutionX(), solver.getResolutionY());
		}
	}
	
	/**
	 * Sets the value inside the specified region
	 * @param target the target buffer
	 * @param flags the flag grid
	 * @param type in the area with that flag, the boundary condition is applied
	 * @param value the value to set
	 */
	public void applyDirichlet(RealGrid target, FlagGrid flags, FlagGrid.CellType type, float value)
	{
		if (solver.is2D()) {
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.getResolutionX() * solver.getResolutionY());
			applyDirichlet2DKernel.Run1NoEvent(solver.clSettings.getClCommandQueue(), ws, 
					target.buffer, flags.buffer, type.value, value,
					solver.getResolutionX(), solver.getResolutionY());
		}
	}
}

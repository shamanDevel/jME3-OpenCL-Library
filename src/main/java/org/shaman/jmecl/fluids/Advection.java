/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.opencl.Kernel;
import com.jme3.opencl.Program;

/**
 * The advection step advects real grids (e.g. density or temperature) or
 * MAC grid (the velocity itself) by the velocity specified as a MAC grid.
 * @author Sebastian
 */
public class Advection {
	private static final String SOURCE_FILE = "org/shaman/jmecl/fluids/Advection.cl";
	
	protected final FluidSolver solver;
	
	protected final RealGrid tmpRealGrid;
	protected final MACGrid tmpMACGrid;
	protected final Kernel semiLagrangeReal2DKernel;
	protected final Kernel semiLagrangeMAC2DKernel;

	public Advection(FluidSolver solver) {
		this.solver = solver;
		
		tmpRealGrid = solver.createRealGrid();
		tmpMACGrid = solver.createMACGrid();
		
		String cacheID = Advection.class.getName();
		Program program = solver.clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			program = solver.clSettings.getClContext().createProgramFromSourceFiles(solver.clSettings.getAssetManager(), SOURCE_FILE);
			program.build();
			solver.clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		semiLagrangeReal2DKernel = program.createKernel("SemiLagrangeReal2D");
		semiLagrangeMAC2DKernel = program.createKernel("SemiLagrangeMAC2D");
	}
	
	public void advect(MACGrid velocity, RealGrid toAdvect, float timestep, int order) {
		if (solver.is2D()) {
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.getResolutionX() * solver.getResolutionY());
			semiLagrangeReal2DKernel.Run1NoEvent(solver.clSettings.getClCommandQueue(), ws, 
					toAdvect.buffer, tmpRealGrid.buffer, velocity.buffer, timestep, solver.getResolutionX(), solver.getResolutionY());
			toAdvect.swap(tmpRealGrid);
		}
	}
	
	public void advect(MACGrid velocity, MACGrid toAdvect, float timestep, int order) {
		if (solver.is2D()) {
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.getResolutionX() * solver.getResolutionY());
			semiLagrangeMAC2DKernel.Run1NoEvent(solver.clSettings.getClCommandQueue(), ws, 
					toAdvect.buffer, tmpMACGrid.buffer, velocity.buffer, timestep, solver.getResolutionX(), solver.getResolutionY());
			toAdvect.swap(tmpMACGrid);
		}
	}
}

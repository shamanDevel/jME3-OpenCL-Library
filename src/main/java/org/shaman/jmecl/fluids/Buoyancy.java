/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.math.Vector2f;
import com.jme3.math.Vector3f;
import com.jme3.math.Vector4f;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.Program;

/**
 *
 * @author Sebastian
 */
public class Buoyancy {
	private static final String SOURCE_FILE = "org/shaman/jmecl/fluids/Buoyancy.cl";
	
	protected final FluidSolver solver;
	
	private final Vector2f tmpVec2;
	private final Vector4f tmpVec4;
	private final Kernel AddSimpleBuoyancy2DKernel;
	private final Kernel AddSimpleBuoyancy3DKernel;

	public Buoyancy(FluidSolver solver) {
		this.solver = solver;
		tmpVec2 = new Vector2f();
		tmpVec4 = new Vector4f();
		
		String cacheID = Buoyancy.class.getName();
		Program program = solver.clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			program = solver.clSettings.getClContext().createProgramFromSourceFiles(solver.clSettings.getAssetManager(), SOURCE_FILE);
			program.build();
			solver.clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		AddSimpleBuoyancy2DKernel = program.createKernel("AddSimpleBuoyancy2D");
		AddSimpleBuoyancy3DKernel = program.createKernel("AddSimpleBuoyancy3D");
	}
	
	public void addBuoynacy(FlagGrid flags, RealGrid density, MACGrid velocity,
			Vector3f gravity, float timestep)
	{
		if (solver.is2D()) {
			tmpVec2.set(-gravity.x, -gravity.y).multLocal(timestep);
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.resolutionX * solver.resolutionY);
			AddSimpleBuoyancy2DKernel.Run1NoEvent(solver.clSettings.getClCommandQueue(), ws,
					flags.buffer, density.buffer, velocity.buffer, 
					tmpVec2, solver.resolutionX, solver.resolutionY);
		} else {
			tmpVec4.set(-gravity.x, -gravity.y, -gravity.z, 0).multLocal(timestep);
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.resolutionX * solver.resolutionY * solver.resolutionZ);
			AddSimpleBuoyancy3DKernel.Run1NoEvent(solver.clSettings.getClCommandQueue(), ws,
					flags.buffer, density.buffer, velocity.buffer, 
					tmpVec4, solver.resolutionX, solver.resolutionY, solver.resolutionZ);
		}
	}
	
}

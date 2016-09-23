/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.Program;
import org.shaman.jmecl.eq.EquationSolver;
import org.shaman.jmecl.eq.JacobiSolver;

/**
 * The pressure projection makes the velocity field divergence free.
 * @author Sebastian
 */
public class PressureProjection {
	private static final String SOURCE_FILE = "org/shaman/jmecl/fluids/PressureProjection.cl";
	private static final boolean DEBUG = false;
	
	protected final FluidSolver solver;
	protected final EquationSolver equationSolver;
	
	protected final Kernel MakeRhs2DKernel;
	protected final Kernel MakeLaplaceMatrix2DKernel;
	protected final Kernel CorrectVelocity2DKernel;
	
	protected boolean needsUpdate;
	protected FlagGrid flagGrid;
	protected boolean preciseBoundaries;
	protected int maxIterations;
	protected float maxError;
	
	protected DebugTools debugTools;

	public PressureProjection(FluidSolver solver, EquationSolver equationSolver) {
		if (equationSolver.getResolutionX() != solver.getResolutionX()
				|| equationSolver.getResolutionY() != solver.getResolutionY()
				|| equationSolver.getResolutionZ() != solver.getResolutionZ()
				|| equationSolver.is2D() != solver.is2D()) {
			throw new IllegalArgumentException("equation solver does not match the fluid solver");
		}
		this.solver = solver;
		this.equationSolver = equationSolver;
		equationSolver.setXToZero();
		needsUpdate = true;
		
		String cacheID = PressureProjection.class.getName();
		Program program = solver.clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			program = solver.clSettings.getClContext().createProgramFromSourceFiles(solver.clSettings.getAssetManager(), SOURCE_FILE);
			program.build();
			solver.clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		MakeRhs2DKernel = program.createKernel("MakeRhs2D");
		MakeLaplaceMatrix2DKernel = program.createKernel("MakeLaplaceMatrix2D");
		CorrectVelocity2DKernel = program.createKernel("CorrectVelocity2D");
		
		if (DEBUG) {
			debugTools = new DebugTools(solver);
		}
	}
	
	protected static EquationSolver createEquationSolver(FluidSolver solver) {
		if (solver.is2D()) {
			return new JacobiSolver(solver.clSettings, solver.resolutionX, solver.resolutionY);
		} else {
			return new JacobiSolver(solver.clSettings, solver.resolutionX, solver.resolutionY, solver.resolutionZ);
		}
	}
	
	public PressureProjection(FluidSolver solver) {
		this(solver, createEquationSolver(solver));
	}
	
	public void setBoundary(FlagGrid flagGrid) {
		this.flagGrid = flagGrid;
		needsUpdate = true;
	}
	
	public void setBoundary(RealGrid solidLevelset, RealGrid liquidLevelset) {
		throw new UnsupportedOperationException("not supported yet");
	}
	
	public void setUse2ndOrderBoundaries(boolean enabled) {
		preciseBoundaries = enabled;
		needsUpdate = true;
	}

	public boolean getUse2ndOrderBoundaries() {
		return preciseBoundaries;
	}
	
	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public float getMaxError() {
		return maxError;
	}

	public void setMaxError(float maxError) {
		this.maxError = maxError;
	}
	
	protected void setupEquation() {
		if (!needsUpdate) {
			return;
		}
		//assemble matrix
		CommandQueue cq = solver.clSettings.getClCommandQueue();
		RealGrid A0 = solver.createRealGrid();
		RealGrid Ain = solver.createRealGrid();
		RealGrid Ajn = solver.createRealGrid();
		RealGrid Aip = solver.createRealGrid();
		RealGrid Ajp = solver.createRealGrid();
		A0.fill(0);
		Ain.fill(0);
		Ajn.fill(0);
		Aip.fill(0);
		Ajp.fill(0);
		if (solver.is2D()) {
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.resolutionX * solver.resolutionY);
			MakeLaplaceMatrix2DKernel.Run1NoEvent(cq, ws, flagGrid.buffer, A0.buffer, Ain.buffer, Aip.buffer, Ajn.buffer, Ajp.buffer, solver.resolutionX, solver.resolutionY);
			equationSolver.setA(A0.getBuffer(), 0, 0, 0);
			equationSolver.setA(Ain.getBuffer(), -1, 0, 0);
			equationSolver.setA(Aip.getBuffer(), 1, 0, 0);
			equationSolver.setA(Ajn.getBuffer(), 0, -1, 0);
			equationSolver.setA(Ajp.getBuffer(), 0, 1, 0);
			
			if (DEBUG) {
				System.out.println("A (0,0,0):");
				debugTools.printGrid2D(A0);
				System.out.println("A (-1,0,0):");
				debugTools.printGrid2D(Ain);
				System.out.println("A (+1,0,0):");
				debugTools.printGrid2D(Aip);
				System.out.println("A (0,-1,0):");
				debugTools.printGrid2D(Ajn);
				System.out.println("A (0,+1,0):");
				debugTools.printGrid2D(Ajp);
			}
			
		} else {
			RealGrid Ak = solver.createRealGrid();
			Ak.fill(0);
			//TODO
			
			Ak.buffer.release();
		}
		equationSolver.assembleMatrix();
		A0.buffer.release();
		Ain.buffer.release();
		Ajn.buffer.release();
		Aip.buffer.release();
		Ajp.buffer.release();
	}
	
	public void debugPrintDivergence(MACGrid velocities) {
		CommandQueue cq = solver.clSettings.getClCommandQueue();
		if (solver.is2D()) {
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.resolutionX * solver.resolutionY);
			
			//setup right hand side
			MakeRhs2DKernel.Run1NoEvent(cq, ws, flagGrid.buffer, equationSolver.getBBuffer(), velocities.buffer, solver.resolutionX, solver.resolutionY);
			System.out.println("Divergence:");
			new DebugTools(solver).printGrid2D(new RealGrid(solver, equationSolver.getBBuffer()));
		}
	}
	
	public void project(MACGrid velocities) {
		setupEquation();
		CommandQueue cq = solver.clSettings.getClCommandQueue();
		if (solver.is2D()) {
			Kernel.WorkSize ws = new Kernel.WorkSize(solver.resolutionX * solver.resolutionY);
			
			if (DEBUG) {
				System.out.println("Velocities:");
				debugTools.printGrid2D(velocities);
			}
			
			//setup right hand side
			MakeRhs2DKernel.Run1NoEvent(cq, ws, flagGrid.buffer, equationSolver.getBBuffer(), velocities.buffer, solver.resolutionX, solver.resolutionY);
			if (DEBUG) {
				System.out.println("Divergence:");
				debugTools.printGrid2D(new RealGrid(solver, equationSolver.getBBuffer()));
			}

			//solve
			equationSolver.setXToZero();
			equationSolver.solve(maxIterations, maxError);
			if (DEBUG) {
				System.out.println("Pressure:");
				debugTools.printGrid2D(new RealGrid(solver, equationSolver.getXBuffer()));
			}

			//correct the velocities
			CorrectVelocity2DKernel.Run1NoEvent(cq, ws, flagGrid.buffer, velocities.buffer, equationSolver.getXBuffer(), solver.resolutionX, solver.resolutionY);
		}
	}
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import org.shaman.jmecl.eq.EquationSolver;

/**
 * The pressure projection makes the velocity field divergence free.
 * @author Sebastian
 */
public class PressureProjection {
	
	protected final FluidSolver solver;
	protected final EquationSolver equationSolver;

	public PressureProjection(FluidSolver solver, EquationSolver equationSolver) {
		this.solver = solver;
		this.equationSolver = equationSolver;
	}
	
	public void setBoundary(FlagGrid flagGrid) {
		throw new UnsupportedOperationException("not supported yet");
	}
	
	public void setBoundary(RealGrid solidLevelset, RealGrid liquidLevelset) {
		throw new UnsupportedOperationException("not supported yet");
	}
	
	public void setUse2ndOrderBoundaries(boolean enabled) {
		throw new UnsupportedOperationException("not supported yet");
	}
	
	public void prpoject(MACGrid velocities) {
		throw new UnsupportedOperationException("not supported yet");
	}
}

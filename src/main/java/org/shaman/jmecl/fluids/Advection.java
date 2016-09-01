/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

/**
 * The advection step advects real grids (e.g. density or temperature) or
 * MAC grid (the velocity itself) by the velocity specified as a MAC grid.
 * @author Sebastian
 */
public class Advection {
	
	protected final FluidSolver solver;

	public Advection(FluidSolver solver) {
		this.solver = solver;
	}
	
	public void advect(MACGrid velocity, RealGrid toAdvect, int order) {
		throw new UnsupportedOperationException("not supported yet");
	}
	
	public void advect(MACGrid velocity, MACGrid toAdvect, int order) {
		throw new UnsupportedOperationException("not supported yet");
	}
}

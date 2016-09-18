/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.opencl.Buffer;
import org.shaman.jmecl.utils.CLBlas;

/**
 * A grid storing reals on the cell centers.
 * @author Sebastian
 */
public class RealGrid extends Grid {
	
	RealGrid(FluidSolver solver) {
		super(solver.clSettings.getClContext().createBuffer(4 * solver.getResolutionX() * solver.getResolutionY() * solver.getResolutionZ()), solver);
	}
	
	RealGrid(FluidSolver solver, Buffer buffer) {
		super(buffer, solver);
	}
	
	public void fill(float value) {
		CLBlas<Float> blas = CLBlas.get(solver.clSettings, Float.class);
		blas.fill(buffer, value);
	}
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.opencl.Buffer;

/**
 *
 * @author Sebastian
 */
abstract class Grid {
	
	protected Buffer buffer;
	protected final FluidSolver solver;

	public Grid(FluidSolver solver) {
		this.solver = solver;
	}

	public Grid(Buffer buffer, FluidSolver solver) {
		this.buffer = buffer;
		this.solver = solver;
	}

	public Buffer getBuffer() {
		return buffer;
	}

	public void setBuffer(Buffer buffer) {
		this.buffer = buffer;
	}

	public FluidSolver getSolver() {
		return solver;
	}

	public void swap(Grid other) {
		assert (this.getClass() == other.getClass());
		Buffer tmp = this.buffer;
		this.buffer = other.buffer;
		other.buffer = tmp;
	}
}

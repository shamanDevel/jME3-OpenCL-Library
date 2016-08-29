/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.eq;

import com.jme3.opencl.Buffer;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.utils.CLBlas;

/**
 * Base class for all sparse linear equation solvers on floats, it solves Ax=b for x.
 * The matrix is defined as a 7-stencil (5-stencil in 2D).
 * @author Sebastian
 */
public abstract class EquationSolver {
	
	protected final OpenCLSettings clSettings;
	protected final int resolutionX;
	protected final int resolutionY;
	protected final int resolutionZ;
	protected final boolean twoD;
	
	protected final Buffer bufX;
	protected final Buffer bufB;

	public EquationSolver(OpenCLSettings clSettings, int resolutionX, int resolutionY, int resolutionZ) {
		this.clSettings = clSettings;
		this.resolutionX = resolutionX;
		this.resolutionY = resolutionY;
		this.resolutionZ = resolutionZ;
		this.twoD = false;
		long size = 4 * resolutionX * resolutionY * resolutionZ;
		this.bufX = clSettings.getClContext().createBuffer(size);
		this.bufB = clSettings.getClContext().createBuffer(size);
	}

	public EquationSolver(OpenCLSettings clSettings, int resolutionX, int resolutionY) {
		this.clSettings = clSettings;
		this.resolutionX = resolutionX;
		this.resolutionY = resolutionY;
		this.resolutionZ = 1;
		this.twoD = true;
		long size = 4 * resolutionX * resolutionY;
		this.bufX = clSettings.getClContext().createBuffer(size);
		this.bufB = clSettings.getClContext().createBuffer(size);
	}

	public boolean is2D() {
		return twoD;
	}

	public int getResolutionX() {
		return resolutionX;
	}

	public int getResolutionY() {
		return resolutionY;
	}

	public int getResolutionZ() {
		return resolutionZ;
	}

	public Buffer getXBuffer() {
		return bufX;
	}

	public Buffer getBBuffer() {
		return bufB;
	}

	public void setXToZero() {
		CLBlas.get(clSettings, Float.class).fill(bufX, 0.0f);
	}
	
	protected void checkStencil(Buffer buf, int stencilX, int stencilY, int stencilZ) {
		if (is2D()) {
			if (stencilZ != 0) {
				throw new IllegalArgumentException("In 2D mode, stencilZ must be zero");
			}
			int stencil = Math.abs(stencilX) + Math.abs(stencilY);
			if (stencil > 1) {
				throw new IllegalArgumentException("as a stencil only (0,0), (1,0),"
						+ " (-1,0), (0,1) and (0,-1) allowed, not ("
						+stencilX+","+stencilY+")");
			}
		} else {
			int stencil = Math.abs(stencilX) + Math.abs(stencilY) + Math.abs(stencilZ);
			if (stencil > 1) {
				throw new IllegalArgumentException("as a stencil only (0,0,0), (1,0,0),"
						+ " (-1,0,0), (0,1,0), (0,-1,0), (0,0,1) and (0,0,-1) allowed, not ("
						+stencilX+","+stencilY+","+stencilZ+")");
			}
		}
		if (buf == null) {
			throw new NullPointerException("buffer is null");
		}
		if (buf.getSize() != resolutionX*resolutionY*resolutionZ*4) {
			throw new IllegalArgumentException("wrong buffer size, expected: "+
					(resolutionX*resolutionY*resolutionZ*4)+", actual: "+buf.getSize());
		}
	}
	public abstract void setA(Buffer buf, int stencilX, int stencilY, int stencilZ);
	
	public void assembleMatrix() {}
	
	public abstract void solve(int maxIteration, float maxError);
}

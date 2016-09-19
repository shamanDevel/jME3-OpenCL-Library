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
 * The matrix elements are specified by {@link #setA(com.jme3.opencl.Buffer, int, int, int) }
 * and the b vector by accessing {@link #getBBuffer() }.
 * The linear system is solved with an invocation of {@link #solve(int, float) }.
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

	/**
	 * Initializes the 3d version of the solver.
	 * @param clSettings the OpenCL settings
	 * @param resolutionX the x-resolution
	 * @param resolutionY the y-resolution
	 * @param resolutionZ the z-resolution
	 */
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

	/**
	 * Initializes the 2d version of the solver.
	 * @param clSettings the OpenCL settings
	 * @param resolutionX the x-resolution
	 * @param resolutionY the y-resolution
	 */
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

	/**
	 * Returns {@code true} if the solver runs in the 2D mode.
	 * @return {@code true} if 2D, {@code false} if 3D.
	 */
	public boolean is2D() {
		return twoD;
	}

	/**
	 * @return the x-resolution
	 */
	public int getResolutionX() {
		return resolutionX;
	}

	/**
	 * @return the y-resolution
	 */
	public int getResolutionY() {
		return resolutionY;
	}

	/**
	 * @return the z-resolution, 1 if 2D
	 */
	public int getResolutionZ() {
		return resolutionZ;
	}

	/**
	 * Accesses the x buffer (the solution).
	 * It is a buffer of size {@code resolutionX*resolutionY*resolutionZ} 
	 * containing floats.
	 * @return the x buffer
	 */
	public Buffer getXBuffer() {
		return bufX;
	}

	/**
	 * Accesses the b buffer (the right hand side of the equation).
	 * It is a buffer of size {@code resolutionX*resolutionY*resolutionZ} 
	 * containing floats.
	 * @return the b buffer
	 */
	public Buffer getBBuffer() {
		return bufB;
	}

	/**
	 * Sets the x buffer to zero.
	 */
	public void setXToZero() {
		CLBlas.get(clSettings, Float.class).fill(bufX, 0.0f);
	}
	
	/**
	 * For the implementations: tests if the arguments to 
	 * {@link #setA(com.jme3.opencl.Buffer, int, int, int) } are valid.
	 * @param buf
	 * @param stencilX
	 * @param stencilY
	 * @param stencilZ 
	 */
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
	/**
	 * Sets a stencil component of the A-matrix.
	 * Allowed stencil values are: (0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0),
	 * (0,0,1) and (0,0,-1). In the 2D mode, the z-coordinate must be zero.
	 * After setting all stencil values, call {@link #assembleMatrix() }.
	 * @param buf the buffer, must contain {@code resolutionX*resolutionY*resolutionZ} 
	 * floats.
	 * @param stencilX the x coordinate of the stencil
	 * @param stencilY the y coordinate of the stencil
	 * @param stencilZ the z coordinate of the stencil
	 */
	public abstract void setA(Buffer buf, int stencilX, int stencilY, int stencilZ);
	
	/**
	 * Assembles the A-matrix.
	 * This must be called after all stencil components are set by
	 * {@link #setA(com.jme3.opencl.Buffer, int, int, int) }.
	 * It is needed e.g. by a multigrid solver to assemble the hierarchy.
	 */
	public void assembleMatrix() {}
	
	/**
	 * Special parameter as {@code maxError} to {@link #solve(int, float) }
	 * indicating that the l2-norm of the residuum should not be computed.
	 * Therefore, no memcopies from the GPU to the CPU are required per iteration,
	 * but the solver will use the full {@code maxIteration} iterations.
	 */
	public static float ERROR_DONT_TEST = -1;
	
	/**
	 * Special parameter as {@code maxError} to {@link #solve(int, float) }
	 * indicating that the l2-norm of the residuum should only be computed at the
	 * end for logging purposes.
	 * Therefore, no memcopies from the GPU to the CPU are required per iteration,
	 * but the solver will use the full {@code maxIteration} iterations.
	 */
	public static float ERROR_ONLY_TEST_AT_THE_END = -2;
	
	/**
	 * Solves the linear system.
	 * Two termination criteria are available: either the solver reaches
	 * the specified maximal number of iterations, or the L2-norm of the residuum
	 * falls below the specified maximal error.
	 * @param maxIteration the maximal number of iterations
	 * @param maxError the maximal error of the residuum,
	 *	or one of the constants {@link #ERROR_DONT_TEST} and {@link #ERROR_ONLY_TEST_AT_THE_END}
	 */
	public abstract void solve(int maxIteration, float maxError);
}

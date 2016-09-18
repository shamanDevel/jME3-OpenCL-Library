/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.eq.EquationSolver;

/**
 *
 * @author Sebastian
 */
public class FluidSolver {
	
	protected final OpenCLSettings clSettings;
	protected final int resolutionX;
	protected final int resolutionY;
	protected final int resolutionZ;
	protected final boolean twoD;
	
	/**
	 * Initializes the 3d version of the fluid solver.
	 * @param clSettings the OpenCL settings
	 * @param resolutionX the x-resolution
	 * @param resolutionY the y-resolution
	 * @param resolutionZ the z-resolution
	 */
	public FluidSolver(OpenCLSettings clSettings, int resolutionX, int resolutionY, int resolutionZ) {
		this.clSettings = clSettings;
		this.resolutionX = resolutionX;
		this.resolutionY = resolutionY;
		this.resolutionZ = resolutionZ;
		this.twoD = false;
	}

	/**
	 * Initializes the 2d version of the fluid solver.
	 * @param clSettings the OpenCL settings
	 * @param resolutionX the x-resolution
	 * @param resolutionY the y-resolution
	 */
	public FluidSolver(OpenCLSettings clSettings, int resolutionX, int resolutionY) {
		this.clSettings = clSettings;
		this.resolutionX = resolutionX;
		this.resolutionY = resolutionY;
		this.resolutionZ = 1;
		this.twoD = true;
	}
	
	/**
	 * Returns {@code true} if the solver runs in the 2D mode.
	 * @return {@code true} if 2D, {@code false} if 3D.
	 */
	public boolean is2D() {
		return twoD;
	}
	
	/**
	 * Returns {@code true} if the solver runs in the 3D mode.
	 * @return {@code true} if 3D, {@code false} if 2D.
	 */
	public boolean is3D() {
		return !twoD;
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
	 * Creates a new real grid linked to this solver.
	 * @return the new real grid
	 */
	public RealGrid createRealGrid() {
		return new RealGrid(this);
	}
	
	/**
	 * Creates a new MAC grid or staggered grid linked to this solver.
	 * @return the new MAC grid
	 */
	public MACGrid createMACGrid() {
		return new MACGrid(this);
	}
	
	/**
	 * Creates a new flag grid linked to this solver.
	 * @return the new flag grid
	 */
	public FlagGrid createFlagGrid() {
		return new FlagGrid(this);
	}
	
}

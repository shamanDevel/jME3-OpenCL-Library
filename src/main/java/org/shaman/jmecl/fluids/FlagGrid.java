/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import org.shaman.jmecl.utils.CLBlas;

/**
 * A grid storing a single byte of flags at the cell centers.
 * @author Sebastian
 */
public class FlagGrid extends Grid {
	
	public static enum CellType { 
		TypeNone(0),
		TypeFluid(1),
		TypeObstacle(2),
		TypeEmpty(4),
		TypeInflow(8),
		TypeOutflow(16),
		TypeOpen(32),
		TypeStick(128),
		TypeReserved(256),
		// 2^10 - 2^14 reserved for moving obstacles
		TypeZeroPressure(1<<15);
				
	public final int value;

	private CellType(int value) {
		this.value = value;
	}
	
	};
	
	FlagGrid(FluidSolver solver) {
		super(solver.clSettings.getClContext().createBuffer(solver.getResolutionX() * solver.getResolutionY() * solver.getResolutionZ()), solver);
	}
	
	public void fill(CellType type) {
		CLBlas<Byte> blas = CLBlas.get(solver.clSettings, Byte.class);
		blas.fill(buffer, (byte) type.value);
	}
	
}

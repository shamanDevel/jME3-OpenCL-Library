/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.math.Vector3f;
import org.shaman.jmecl.utils.CLBlas;

/**
 * A grid that stores velocity components on the cell faces
 * @author Sebastian
 */
public class MACGrid extends Grid {
	
	MACGrid(FluidSolver solver) {
		super(solver);
		long sizeX = (solver.getResolutionX()+1) * solver.getResolutionY() * solver.getResolutionZ();
		long sizeY = solver.getResolutionX() * (solver.getResolutionY()+1) * solver.getResolutionZ();
		long sizeZ = solver.getResolutionX() * solver.getResolutionY() * (solver.getResolutionZ()+1);
		buffer = solver.clSettings.getClContext().createBuffer(4 * (sizeX + sizeY + (solver.is2D() ? 0 : sizeZ)));
	}

	public void fill(Vector3f v) {
		CLBlas<Float> blas = CLBlas.get(solver.clSettings, Float.class);
		long sizeX = (solver.getResolutionX()+1) * solver.getResolutionY() * solver.getResolutionZ();
		long sizeY = solver.getResolutionX() * (solver.getResolutionY()+1) * solver.getResolutionZ();
		long sizeZ = solver.getResolutionX() * solver.getResolutionY() * (solver.getResolutionZ()+1);
		blas.fill(buffer, v.x, sizeX);
		blas.fill(buffer, v.y, sizeY, sizeX, 1);
		if (solver.is3D()) {
			blas.fill(buffer, v.z, sizeZ, sizeX + sizeY, 1);
		}
	}
}

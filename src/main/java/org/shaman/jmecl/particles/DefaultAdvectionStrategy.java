/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.opencl.Buffer;

/**
 * A default advection strategy.
 *
 * Particles can have:
 * <ul>
 * <li>a position</li>
 * <li>a velocity</li>
 * <li>an acceleration, or a global acceleration</li>
 * <li>a mass, or a global mass</li>
 * <li>a color</li>
 * </ul>
 *
 */
public class DefaultAdvectionStrategy implements AdvectionStrategy {

	@Override
	public void init(ParticleController controller) {
//		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public void resized(int newSize) {
//		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public void advect(float tpf, int offset, int count, Buffer deletionBuffer) {
//		throw new UnsupportedOperationException("Not supported yet.");
	}
	
}

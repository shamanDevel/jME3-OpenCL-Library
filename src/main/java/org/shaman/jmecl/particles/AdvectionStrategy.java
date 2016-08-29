/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.opencl.Buffer;

/**
 *
 * @author Sebastian Weiss
 */
public interface AdvectionStrategy {
	
	void init(ParticleController controller);
	
	void resized(int newSize);
	
	void advect(float tpf, int count, Buffer deletionBuffer);
	
	
}

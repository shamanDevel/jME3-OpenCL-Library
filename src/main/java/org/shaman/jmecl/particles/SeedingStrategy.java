/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

/**
 *
 * @author Sebastian Weiss
 */
public interface SeedingStrategy {
	
	void init(ParticleController controller);
	
	void resized(int newSize);
	
	int getNewParticlesCount(float tpf);
	
	void initNewParticles(float tpf, int offset, int count);
	
}

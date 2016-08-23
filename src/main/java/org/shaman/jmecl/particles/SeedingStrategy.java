/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

/**
 * Interface for particle seeds.
 * @author Sebastian Weiss
 */
public interface SeedingStrategy {
	/**
	 * Initializes the seeding strategy
	 * @param controller the particle controller
	 */
	void init(ParticleController controller);
	
	/**
	 * This method is called when the capacity is resized.
	 * Use it if you have custom buffers allocated
	 * @param newSize the new capacity of the particle buffers
	 */
	void resized(int newSize);
	
	/**
	 * Requests how many new particles should be spawned.
	 * @param tpf the time since the last frame
	 * @return the number of new particles to add
	 */
	int getNewParticlesCount(float tpf);
	
	/**
	 * Initializes the new particles
	 * @param tpf the time since the last frame
	 * @param offset the offset into the buffers provided by the particle controller
	 * @param count the maximal number of particles that can be created
	 * @return the actual number of particles created, must not be greater than {@code count}
	 */
	int initNewParticles(float tpf, int offset, int count);
	
}

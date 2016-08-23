/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A seeding strategy that combines multiple seeding strategies together.
 * By this class, multiple seeds can be added to a particle controller.
 * @author Sebastian Weiss
 */
public class CombiningSeedingStrategy implements SeedingStrategy {

	private ParticleController controller = null;
	private int size = -1;
	private final List<SeedingStrategy> strategies;

	public CombiningSeedingStrategy() {
		strategies = new ArrayList<>();
	}

	public CombiningSeedingStrategy(List<? extends SeedingStrategy> strategies) {
		this.strategies = new ArrayList<>(strategies);
	}
	
	public CombiningSeedingStrategy(SeedingStrategy... strategies) {
		this.strategies = new ArrayList<>(Arrays.asList(strategies));
	}
	
	public void addStrategy(SeedingStrategy s) {
		strategies.add(s);
		if (controller != null) {
			s.init(controller);
		}
		if (size != -1) {
			s.resized(size);
		}
	}
	
	public boolean removeStrategy(SeedingStrategy s) {
		return strategies.remove(s);
	}
	
	@Override
	public void init(ParticleController controller) {
		this.controller = controller;
		for (SeedingStrategy s : strategies) {
			s.init(controller);
		}
	}

	@Override
	public void resized(int newSize) {
		this.size = newSize;
		for (SeedingStrategy s : strategies) {
			s.resized(newSize);
		}
	}

	@Override
	public int getNewParticlesCount(float tpf) {
		int count = 0;
		for (SeedingStrategy s : strategies) {
			count += s.getNewParticlesCount(tpf);
		}
		return count;
	}

	@Override
	public int initNewParticles(float tpf, int offset, int count) {
		int created = 0;
		for (SeedingStrategy s : strategies) {
			int c = s.initNewParticles(tpf, offset, count);
			created += c;
			count -= c;
			offset += c;
		}
		return created;
	}
	
}

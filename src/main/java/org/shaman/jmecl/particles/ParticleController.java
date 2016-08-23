/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.opencl.Buffer;
import com.jme3.opencl.MemoryAccess;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.ViewPort;
import com.jme3.scene.VertexBuffer;
import com.jme3.scene.control.AbstractControl;
import java.util.EnumMap;
import java.util.Map;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.rendering.ParticleRenderer;
import org.shaman.jmecl.sorting.RadixSort;
import org.shaman.jmecl.utils.SharedBuffer;

/**
 *
 * @author Sebastian Weiss
 */
public class ParticleController extends AbstractControl {
	private final ParticleRenderer renderer;
	
	private OpenCLSettings clSettings;
	private EnumMap<VertexBuffer.Type, SharedBuffer> buffers;
	private SeedingStrategy seedingStrategy;
	private AdvectionStrategy advectionStrategy;
	
	private Buffer deletionBuffer;
	private Buffer indexBuffer;
	private Buffer lastIndexBuffer;
	private RadixSort radixSort;

	public ParticleController(ParticleRenderer renderer) {
		this.renderer = renderer;
		buffers = new EnumMap<>(VertexBuffer.Type.class);
	}

	public void addBuffer(VertexBuffer buffer) {
		buffers.put(buffer.getBufferType(), new SharedBuffer(buffer));
	}

	public SeedingStrategy getSeedingStrategy() {
		return seedingStrategy;
	}

	public void setSeedingStrategy(SeedingStrategy seedingStrategy) {
		this.seedingStrategy = seedingStrategy;
	}

	public AdvectionStrategy getAdvectionStrategy() {
		return advectionStrategy;
	}

	public void setAdvectionStrategy(AdvectionStrategy advectionStrategy) {
		this.advectionStrategy = advectionStrategy;
	}
	
	public void init(RenderManager renderManager, OpenCLSettings clSettings) {
		this.clSettings = clSettings;
		
		//init buffers
		for (SharedBuffer b : buffers.values()) {
			b.initialize(renderManager, clSettings.getClContext(), MemoryAccess.READ_WRITE);
		}
		int capacity = renderer.getCapacity();
		deletionBuffer = clSettings.getClContext().createBuffer(capacity * 4, MemoryAccess.READ_WRITE).register();
		indexBuffer = clSettings.getClContext().createBuffer(capacity * 4, MemoryAccess.READ_WRITE).register();
		lastIndexBuffer = clSettings.getClContext().createBuffer(4).register();
		
		seedingStrategy.init(this);
		seedingStrategy.resized(capacity);
		advectionStrategy.init(this);
		advectionStrategy.resized(capacity);
		
		radixSort = new RadixSort(clSettings);
		
	}
	
	public SharedBuffer getBuffer(VertexBuffer.Type type) {
		return buffers.get(type);
	}
	
	@Override
	protected void controlUpdate(float f) {
		//seed
		int toCreate = seedingStrategy.getNewParticlesCount(f);
		int oldCount = renderer.getParticleCount();
		int requiredCount = toCreate + oldCount;
		int oldCapacity = renderer.getCapacity();
		if (requiredCount > oldCapacity) {
			//resize
			int newSize = Math.max(requiredCount, oldCapacity * 2);
			for (SharedBuffer b : buffers.values()) {
				b.resize(newSize, clSettings.getClCommandQueue());
			}
			seedingStrategy.resized(newSize);
			advectionStrategy.resized(newSize);
			deletionBuffer.release();
			indexBuffer.release();
			deletionBuffer = clSettings.getClContext().createBuffer(newSize * 4, MemoryAccess.READ_WRITE).register();
			indexBuffer = clSettings.getClContext().createBuffer(newSize * 4, MemoryAccess.READ_WRITE).register();
		}
		renderer.setParticleCount(requiredCount);
		seedingStrategy.initNewParticles(f, oldCount, toCreate);
		
		//advect
		advectionStrategy.advect(f, 0, requiredCount, deletionBuffer);
		
		//sort to delete old particles
		radixSort.setEndBit(4);
		radixSort.sort(deletionBuffer, indexBuffer);
	}

	@Override
	protected void controlRender(RenderManager rm, ViewPort vp) {}
	
}

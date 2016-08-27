/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.app.SimpleApplication;
import com.jme3.opencl.Buffer;
import com.jme3.opencl.MemoryAccess;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.ViewPort;
import com.jme3.renderer.opengl.GLRenderer;
import com.jme3.scene.VertexBuffer;
import com.jme3.scene.control.AbstractControl;
import com.jme3.util.BufferUtils;
import java.util.EnumMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.rendering.ParticleRenderer;
import org.shaman.jmecl.sorting.RadixSort;
import org.shaman.jmecl.utils.SharedBuffer;

/**
 * Particle controller,
 * the buffers are accessed by their {@link VertexBuffer.Type}.
 * Default mapping (expected by the default strategies):
 * <ul>
 *  <li>Position -> float4 (pos.x, pos.y, pos.z, time)</li>
 *  <li>TexCoord -> unused</li>
 *  <li>TexCoord2 -> float4 (vel.x, vel.y, vel.z, density)</li>
 * </ul>
 * @author Sebastian Weiss
 */
public class ParticleController extends AbstractControl {
	private static final Logger LOG = Logger.getLogger(ParticleController.class.getName());
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
	
	/**
	 * Initializes the default buffers as expected by the default strategies.
	 * See the class description for the buffers.
	 * Calls {@link #addBuffer(com.jme3.scene.VertexBuffer) } multiple times
	 * @param initialCapacity the initial capacity
	 */
	public void initDefaultBuffers(int initialCapacity) {
		VertexBuffer vb;
		
		vb = new VertexBuffer(VertexBuffer.Type.Position);
		vb.setupData(VertexBuffer.Usage.Dynamic, 4, VertexBuffer.Format.Float, BufferUtils.createFloatBuffer(initialCapacity * 4));
		addBuffer(vb);
		
		vb = new VertexBuffer(VertexBuffer.Type.TexCoord2);
		vb.setupData(VertexBuffer.Usage.Dynamic, 4, VertexBuffer.Format.Float, BufferUtils.createFloatBuffer(initialCapacity * 4));
		addBuffer(vb);
	}

	public void addBuffer(VertexBuffer buffer) {
		buffers.put(buffer.getBufferType(), new SharedBuffer(buffer));
		renderer.getMesh().setBuffer(buffer);
		LOG.log(Level.INFO, "buffer {0} added to particle controller", buffer);
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
		LOG.info("particle controller initialized");
	}

	public OpenCLSettings getCLSettings() {
		return clSettings;
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
				renderer.getMesh().clearBuffer(b.getJMEBuffer().getBufferType());
				renderer.getMesh().setBuffer(b.getJMEBuffer());
			}
			seedingStrategy.resized(newSize);
			advectionStrategy.resized(newSize);
			deletionBuffer.release();
			indexBuffer.release();
			deletionBuffer = clSettings.getClContext().createBuffer(newSize * 4, MemoryAccess.READ_WRITE).register();
			indexBuffer = clSettings.getClContext().createBuffer(newSize * 4, MemoryAccess.READ_WRITE).register();
			return;
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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.app.SimpleApplication;
import com.jme3.opencl.Buffer;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.MappingAccess;
import com.jme3.opencl.MemoryAccess;
import com.jme3.opencl.Program;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.ViewPort;
import com.jme3.renderer.opengl.GLRenderer;
import com.jme3.scene.VertexBuffer;
import com.jme3.scene.control.AbstractControl;
import com.jme3.util.BufferUtils;
import java.nio.ByteBuffer;
import java.util.EnumMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.sorting.RadixSort;
import org.shaman.jmecl.utils.CLBlas;
import org.shaman.jmecl.utils.SharedBuffer;

/**
 * Particle controller,
 * the buffers are accessed by their {@link VertexBuffer.Type}.
 * Default mapping (expected by the default strategies):
 * <ul>
 *  <li>Position -> float4 (pos.x, pos.y, pos.z, density)</li>
 *  <li>TexCoord -> unused</li>
 *  <li>TexCoord2 -> float4 (vel.x, vel.y, vel.z, time)</li>
 *  <li>TexCoord3 -> float1 (temperature)</li>
 * </ul>
 * @author Sebastian Weiss
 */
public class ParticleController extends AbstractControl {
	private static final Logger LOG = Logger.getLogger(ParticleController.class.getName());
	private static final String SOURCE_FILE = "org/shaman/jmecl/particles/ParticleController.cl";
	private final ParticleRenderer renderer;
	
	private OpenCLSettings clSettings;
	private EnumMap<VertexBuffer.Type, SharedBuffer> buffers;
	private SeedingStrategy seedingStrategy;
	private AdvectionStrategy advectionStrategy;
	
	private Buffer deletionBuffer;
	private Buffer indexBuffer;
	private Buffer reorderTempBuffer;
	private Buffer deletionIndexBuffer;
	private Kernel findFirstOneKernel;
	private RadixSort radixSort;
	
	private CLBlas<Integer> blasInt;
	private CLBlas<Float> blasFloat;

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
		
		vb = new VertexBuffer(VertexBuffer.Type.TexCoord3);
		vb.setupData(VertexBuffer.Usage.Dynamic, 1, VertexBuffer.Format.Float, BufferUtils.createFloatBuffer(initialCapacity * 1));
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
		reorderTempBuffer = clSettings.getClContext().createBuffer(capacity * 4 * 4, MemoryAccess.READ_WRITE).register();
		
		deletionIndexBuffer = clSettings.getClContext().createBuffer(4, MemoryAccess.READ_WRITE).register();
		String cacheID = ParticleController.class.getName();
		Program program = clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			program = clSettings.getClContext().createProgramFromSourceFiles(clSettings.getAssetManager(), SOURCE_FILE);
			program.build();
			clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		findFirstOneKernel = program.createKernel("FindFirstOne");
		
		seedingStrategy.init(this);
		seedingStrategy.resized(capacity);
		advectionStrategy.init(this);
		advectionStrategy.resized(capacity);
		
		radixSort = new RadixSort(clSettings);
		blasInt = CLBlas.get(clSettings, Integer.class);
		blasFloat = CLBlas.get(clSettings, Float.class);
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
			reorderTempBuffer.release();
			deletionBuffer = clSettings.getClContext().createBuffer(newSize * 4, MemoryAccess.READ_WRITE).register();
			indexBuffer = clSettings.getClContext().createBuffer(newSize * 4, MemoryAccess.READ_WRITE).register();
			reorderTempBuffer = clSettings.getClContext().createBuffer(newSize * 4 * 4, MemoryAccess.READ_WRITE).register();
//			return;
		}
		for (SharedBuffer b : buffers.values()) {
			b.aquireCLBuffer(clSettings.getClCommandQueue());
		}
		seedingStrategy.initNewParticles(f, oldCount, toCreate);
		
		//advect
		blasInt.fill(deletionBuffer, 0, requiredCount);
		advectionStrategy.advect(f, requiredCount, deletionBuffer);
		
		//delete old particles
		blasInt.fillIndices(indexBuffer, 0, 1, requiredCount);
		radixSort.setEndBit(4);
		/*
		ByteBuffer mappedBuffer = deletionBuffer.map(clSettings.getClCommandQueue(), MappingAccess.MAP_READ_ONLY);
		System.out.print("buf:");
		for (int i=0; i<requiredCount; ++i) {
			System.out.print(" " + mappedBuffer.getInt());
		}
		System.out.println();
		deletionBuffer.unmap(clSettings.getClCommandQueue(), mappedBuffer);
		*/
		radixSort.sort(deletionBuffer, indexBuffer, requiredCount);
		/*
		mappedBuffer = indexBuffer.map(clSettings.getClCommandQueue(), MappingAccess.MAP_READ_ONLY);
		System.out.print("indices:");
		for (int i=0; i<requiredCount; ++i) {
			System.out.print(" " + mappedBuffer.getInt());
		}
		System.out.println();
		indexBuffer.unmap(clSettings.getClCommandQueue(), mappedBuffer);
		System.out.println();
		*/
		//Buffer blockScan = radixSort.getInternalBlockScan();
		blasInt.fill(deletionIndexBuffer, requiredCount);
		findFirstOneKernel.Run1NoEvent(clSettings.getClCommandQueue(), new Kernel.WorkSize(requiredCount), deletionBuffer, deletionIndexBuffer);
		Buffer.AsyncMapping mapping = deletionIndexBuffer.mapAsync(clSettings.getClCommandQueue(), 4, 0, MappingAccess.MAP_READ_ONLY);
		for (SharedBuffer b : buffers.values()) {
			int components = b.getJMEBuffer().getNumComponents();
			Buffer buf = b.getCLBuffer();
			blasFloat.reorder(indexBuffer, buf, reorderTempBuffer, components, requiredCount);
			reorderTempBuffer.copyToAsync(clSettings.getClCommandQueue(), buf, requiredCount * 4 * components);
		}
		mapping.event.waitForFinished();
		int newCount = mapping.buffer.getInt();
		deletionIndexBuffer.unmap(clSettings.getClCommandQueue(), mapping.buffer);
		
		for (SharedBuffer b : buffers.values()) {
			b.releaseCLBuffer(clSettings.getClCommandQueue());
		}
		renderer.setParticleCount(newCount);
		System.out.println("new particles: "+toCreate+", deleted particles: "+(requiredCount-newCount)+", active particles: "+newCount);
		
	}

	@Override
	protected void controlRender(RenderManager rm, ViewPort vp) {}
	
}

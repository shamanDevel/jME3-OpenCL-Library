/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.math.Vector3f;
import com.jme3.opencl.Buffer;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.Program;
import com.jme3.scene.VertexBuffer;
import com.jme3.util.BufferUtils;
import com.jme3.util.TempVars;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.shaman.jmecl.OpenCLSettings;

/**
 *
 * @author Sebastian Weiss
 */
public class ShapeSeedingStrategy implements SeedingStrategy {
	private static final int RANDOM_SEED_COUNT = 2048;
	private static final String SOURCE_FILE = "org/shaman/jmecl/particles/ShapeSeedingStrategy.cl";
	private static final Map<OpenCLSettings, Kernels> kernelMap = new HashMap<>();
	
	private ParticleController controller;
	private Kernels kernels;
	
	private float particlesPerSecond;
	private Vector3f initialVelocity;
	private float velocityVariation;
	private float initialDensity;
	private float densityVariation;
	private Shape shape;

	public float getParticlesPerSecond() {
		return particlesPerSecond;
	}

	public void setParticlesPerSecond(float particlesPerSecond) {
		this.particlesPerSecond = particlesPerSecond;
	}

	public Vector3f getInitialVelocity() {
		return initialVelocity;
	}

	public void setInitialVelocity(Vector3f initialVelocity) {
		if (initialVelocity == null) {
			throw new NullPointerException("initial velocity is null");
		}
		this.initialVelocity = initialVelocity;
	}

	public float getVelocityVariation() {
		return velocityVariation;
	}

	public void setVelocityVariation(float velocityVariation) {
		if (velocityVariation < 0) {
			throw new IllegalArgumentException("velocity variation must not be negative");
		}
		this.velocityVariation = velocityVariation;
	}

	public float getInitialDensity() {
		return initialDensity;
	}

	public void setInitialDensity(float initialDensity) {
		if (initialDensity <= 0) {
			throw new IllegalArgumentException("density variation must be positive");
		}
		this.initialDensity = initialDensity;
	}

	public float getDensityVariation() {
		return densityVariation;
	}

	public void setDensityVariation(float densityVariation) {
		if (densityVariation < 0) {
			throw new IllegalArgumentException("density variation must not be negative");
		}
		this.densityVariation = densityVariation;
	}

	public Shape getShape() {
		return shape;
	}

	public void setShape(Shape shape) {
		if (shape == null) {
			throw new NullPointerException("null shape");
		}
		if (!((shape instanceof Point) || (shape instanceof Sphere) || (shape instanceof Box))) {
			throw new IllegalArgumentException("only Point, Sphere and Box are supported, not "+shape);
		}
		this.shape = shape;
	}

	@Override
	public void init(ParticleController controller) {
		this.controller = controller;
		OpenCLSettings settings = controller.getCLSettings();
		synchronized(kernelMap) {
			kernels = kernelMap.get(settings);
			if (kernels == null) {
				//create it
				kernels = new Kernels();
				kernels.clQueue = settings.getClCommandQueue();
				
				kernels.seedCount = RANDOM_SEED_COUNT;
				kernels.seeds = settings.getClContext().createBuffer(kernels.seedCount * 8);
				Random initRandom = new Random();
				long[] seeds = new long[kernels.seedCount];
				for (int i = 0; i < kernels.seedCount; ++i) {
					seeds[i] = initRandom.nextLong();
						seeds[i] = (seeds[i] ^ 0x5DEECE66DL) & ((1L << 48) - 1);
				}
				ByteBuffer tmpByteBuffer = BufferUtils.createByteBuffer(8 * kernels.seedCount);
				tmpByteBuffer.asLongBuffer().put(seeds);
				kernels.seeds.writeAsync(settings.getClCommandQueue(), tmpByteBuffer).release();
				
				String cacheID = ShapeSeedingStrategy.class.getName();
				Program program = settings.getProgramCache().loadFromCache(cacheID);
				if (program == null) {
					program = settings.getClContext().createProgramFromSourceFiles(settings.getAssetManager(), SOURCE_FILE);
					program.build();
					settings.getProgramCache().saveToCache(cacheID, program);
				}
				program.register();
				kernels.InitParticlesKernel = program.createKernel("InitParticles").register();
				kernels.SeedPointKernel = program.createKernel("SeedPoint").register();
				kernels.SeedSphereKernel = program.createKernel("SeedSphere").register();
				kernels.SeedBoxKernel = program.createKernel("SeedBox").register();
				
				kernelMap.put(settings, kernels);
			}
		}
	}

	@Override
	public void resized(int newSize) {
		
	}

	@Override
	public int getNewParticlesCount(float tpf) {
		//return (int) Math.ceil(tpf * particlesPerSecond);
		return 1;
	}

	@Override
	public int initNewParticles(float tpf, int offset, int count) {
		int actualCount = Math.min(count, getNewParticlesCount(tpf));
		
		Kernel.WorkSize ws = new Kernel.WorkSize();
		Buffer positionBuffer = controller.getBuffer(VertexBuffer.Type.Position).getCLBuffer();
		Buffer velocityBuffer = controller.getBuffer(VertexBuffer.Type.TexCoord2).getCLBuffer();
		
		TempVars vars = TempVars.get();
		try {
			
		kernels.clQueue.finish();
		positionBuffer.acquireBufferForSharingNoEvent(kernels.clQueue);
		kernels.clQueue.finish();
		velocityBuffer.acquireBufferForSharingNoEvent(kernels.clQueue);
		kernels.clQueue.finish();
		
		//seed particles
		int offset2 = offset;
		int left = actualCount;
		while (left > 0) {
			int s = Math.min(left, kernels.seedCount);
			ws.set(1, s, 1, 1);
			if (shape instanceof Point) {
				vars.vect4f1.setX(((Point) shape).position.x);
				vars.vect4f1.setY(((Point) shape).position.y);
				vars.vect4f1.setZ(((Point) shape).position.z);
				kernels.SeedPointKernel.Run1NoEvent(kernels.clQueue, ws, offset2, positionBuffer, 
						vars.vect4f1, tpf, kernels.seeds);
			} else if (shape instanceof Sphere) {
				vars.vect4f1.setX(((Sphere) shape).center.x);
				vars.vect4f1.setY(((Sphere) shape).center.y);
				vars.vect4f1.setZ(((Sphere) shape).center.z);
				kernels.SeedSphereKernel.Run1NoEvent(kernels.clQueue, ws, offset2, positionBuffer, 
						vars.vect4f1, ((Sphere) shape).radius, tpf, kernels.seeds);
			} else if (shape instanceof Box) {
				vars.vect4f1.setX(((Box) shape).min.x);
				vars.vect4f1.setY(((Box) shape).min.y);
				vars.vect4f1.setZ(((Box) shape).min.z);
				vars.vect4f2.setX(((Box) shape).max.x);
				vars.vect4f2.setY(((Box) shape).max.y);
				vars.vect4f2.setZ(((Box) shape).max.z);
				kernels.SeedBoxKernel.Run1NoEvent(kernels.clQueue, ws, offset2, positionBuffer, 
						vars.vect4f1, vars.vect4f2, tpf, kernels.seeds);
			}
			left -= s;
			offset2 += s;
		}
		
		//init particles
		ws.set(1, actualCount, 1, 1);
		vars.vect4f1.setX(initialVelocity.x);
		vars.vect4f1.setY(initialVelocity.y);
		vars.vect4f1.setZ(initialVelocity.z);
		kernels.InitParticlesKernel.Run1NoEvent(kernels.clQueue, ws, offset, velocityBuffer,
				vars.vect4f1, velocityVariation, initialDensity, densityVariation, kernels.seeds);
		
		positionBuffer.releaseBufferForSharingNoEvent(kernels.clQueue);
		velocityBuffer.releaseBufferForSharingNoEvent(kernels.clQueue);
		kernels.clQueue.finish();
		
		} finally {
			vars.release();
		}
		
		return actualCount;
	}
	
	public static interface Shape {
		//marker interface
	}
	public static class Point implements Shape {
		private final Vector3f position;

		public Point(Vector3f position) {
			this.position = position;
		}

		public Vector3f getPosition() {
			return position;
		}

		@Override
		public String toString() {
			return "PointShape{" + "position=" + position + '}';
		}
		
	}
	public static class Sphere implements Shape {
		private final Vector3f center;
		private final float radius;

		public Sphere(Vector3f center, float radius) {
			this.center = center;
			this.radius = radius;
		}

		public Vector3f getCenter() {
			return center;
		}

		public float getRadius() {
			return radius;
		}

		@Override
		public String toString() {
			return "SphereShape{" + "center=" + center + ", radius=" + radius + '}';
		}
		
	}
	public static class Box implements Shape {
		private final Vector3f min;
		private final Vector3f max;

		public Box(Vector3f min, Vector3f max) {
			this.min = min;
			this.max = max;
		}

		public Vector3f getMin() {
			return min;
		}

		public Vector3f getMax() {
			return max;
		}

		@Override
		public String toString() {
			return "BoxShape{" + "min=" + min + ", max=" + max + '}';
		}
		
	}
	
	private static class Kernels {
		private Kernel SeedPointKernel;
		private Kernel SeedSphereKernel;
		private Kernel SeedBoxKernel;
		private Kernel InitParticlesKernel;
		private Buffer seeds;
		private int seedCount;
		private CommandQueue clQueue;
	}
}

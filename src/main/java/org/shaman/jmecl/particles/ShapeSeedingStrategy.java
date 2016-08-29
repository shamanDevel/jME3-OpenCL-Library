/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.particles;

import com.jme3.math.Vector3f;
import com.jme3.math.Vector4f;
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
	private final Vector4f initialVelocity = new Vector4f();
	private float velocityVariation;
	private float initialDensity;
	private float densityVariation;
	private float initialTemperature;
	private float temperatureVariation;
	private Shape shape;

	public float getParticlesPerSecond() {
		return particlesPerSecond;
	}

	public void setParticlesPerSecond(float particlesPerSecond) {
		this.particlesPerSecond = particlesPerSecond;
	}

	public Vector3f getInitialVelocity() {
		return new Vector3f(initialVelocity.x, initialVelocity.y, initialVelocity.z);
	}

	public void setInitialVelocity(Vector3f initialVelocity) {
		if (initialVelocity == null) {
			throw new NullPointerException("initial velocity is null");
		}
		this.initialVelocity.x = initialVelocity.x;
		this.initialVelocity.y = initialVelocity.y;
		this.initialVelocity.z = initialVelocity.z;
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

	public float getInitialTemperature() {
		return initialTemperature;
	}

	public void setInitialTemperature(float initialTemperature) {
		this.initialTemperature = initialTemperature;
	}

	public float getTemperatureVariation() {
		return temperatureVariation;
	}

	public void setTemperatureVariation(float temperatureVariation) {
		if (temperatureVariation < 0) {
			throw new IllegalArgumentException("temperature variation must not be negative");
		}
		this.temperatureVariation = temperatureVariation;
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
		return (int) Math.ceil(tpf * particlesPerSecond);
	}

	@Override
	public int initNewParticles(float tpf, int offset, int count) {
		int actualCount = Math.min(count, getNewParticlesCount(tpf));
		
		Kernel.WorkSize ws = new Kernel.WorkSize();
		Buffer positionBuffer = controller.getBuffer(VertexBuffer.Type.Position).getCLBuffer();
		Buffer velocityBuffer = controller.getBuffer(VertexBuffer.Type.TexCoord2).getCLBuffer();
		Buffer temperatureBuffer = controller.getBuffer(VertexBuffer.Type.TexCoord3).getCLBuffer();
		
		//seed particles
		int offset2 = offset;
		int left = actualCount;
		while (left > 0) {
			int s = Math.min(left, kernels.seedCount);
			ws.set(1, s, 1, 1);
			if (shape instanceof Point) {
				kernels.SeedPointKernel.Run1NoEvent(kernels.clQueue, ws, offset2, positionBuffer, 
						((Point) shape).position, tpf, kernels.seeds);
			} else if (shape instanceof Sphere) {
				kernels.SeedSphereKernel.Run1NoEvent(kernels.clQueue, ws, offset2, positionBuffer, 
						((Sphere) shape).centerAndRadius, tpf, kernels.seeds);
			} else if (shape instanceof Box) {
				kernels.SeedBoxKernel.Run1NoEvent(kernels.clQueue, ws, offset2, positionBuffer, 
						((Box) shape).min, ((Box) shape).max, tpf, kernels.seeds);
			}
			left -= s;
			offset2 += s;
		}
		
		//init particles
		ws.set(1, actualCount, 1, 1);
		kernels.InitParticlesKernel.Run1NoEvent(kernels.clQueue, ws, offset, 
				positionBuffer, velocityBuffer, temperatureBuffer,
				initialVelocity, velocityVariation, initialDensity, densityVariation, initialTemperature, temperatureVariation,
				kernels.seeds);

		return actualCount;
	}
	
	public static interface Shape {
		//marker interface
	}
	public static class Point implements Shape {
		private final Vector4f position;

		public Point(Vector3f position) {
			this.position = new Vector4f(position.x, position.y, position.z, 0);
		}

		public Vector3f getPosition() {
			return new Vector3f(position.x, position.y, position.z);
		}

		@Override
		public String toString() {
			return "PointShape{" + "position=" + position + '}';
		}
		
	}
	public static class Sphere implements Shape {
		private final Vector4f centerAndRadius;

		public Sphere(Vector3f center, float radius) {
			this.centerAndRadius = new Vector4f(center.x, center.y, center.z, radius);
		}

		public Vector3f getCenter() {
			return new Vector3f(centerAndRadius.x, centerAndRadius.y, centerAndRadius.z);
		}
		
		public float getRadius() {
			return centerAndRadius.w;
		}

		@Override
		public String toString() {
			return "SphereShape{" + "center=" + getCenter() + ", radius=" + getRadius() + '}';
		}
		
	}
	public static class Box implements Shape {
		private final Vector4f min;
		private final Vector4f max;

		public Box(Vector3f min, Vector3f max) {
			this.min = new Vector4f(min.x, min.y, min.z, 0);
			this.max = new Vector4f(max.x, max.y, max.z, 0);
		}

		public Vector3f getMin() {
			return new Vector3f(min.x, min.y, min.z);
		}

		public Vector3f getMax() {
			return new Vector3f(max.x, max.y, max.z);
		}

		@Override
		public String toString() {
			return "BoxShape{" + "min=" + getMin() + ", max=" + getMax() + '}';
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

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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.shaman.jmecl.OpenCLSettings;

/**
 * A default advection strategy.
 * <p>
 * Particles can have:
 * <ul>
 * <li>a position</li>
 * <li>a velocity</li>
 * <li>an acceleration, or a global acceleration</li>
 * <li>a mass, or a global mass</li>
 * <li>a color</li>
 * </ul>
 * <p>
 * Deletion by (multiple can be specified):
 * <ul>
 * <li>axis aligned bounding box</li>
 * <li>bounding sphere</li>
 * <li>lower density threshold</li>
 * </ul>
 * <p>
 * Advection equation: <br>
 * <code> v = v + dt * (alpha*density*gravity - beta * temperature * gravity + velocity_field(x)/density </code> <br>
 * <code> x = x + dt * v </code>
 * <p>
 * Exponential decay: <br>
 * <code> temperature = temperature - dt * temperature * lambda </code> <br>
 * <code> density = density - dt * temperature * mu </code> <br>
 * It must hold that <code>dt * lambda < 1</code>, <code>dt * my < 1</code> for the largest possible timestep <code>dt</code>
 */
public class DefaultAdvectionStrategy implements AdvectionStrategy {

	private static final String SOURCE_FILE = "org/shaman/jmecl/particles/DefaultAdvectionStrategy.cl";
	private static final Map<OpenCLSettings, Kernels> kernelMap = new HashMap<>();
	
	private ParticleController controller;
	private Kernels kernels;
	
	private final ArrayList<DeletionCondition> deletionConditions;
	private float alpha;
	private float beta;
	private Vector4f gravity;
	private float lambda;
	private float mu;

	public DefaultAdvectionStrategy() {
		this.deletionConditions = new ArrayList<>();
		this.gravity = new Vector4f();
	}
	
	public void addDeletionCondition(DeletionCondition condition) {
		deletionConditions.add(condition);
	}
	
	public boolean removeDeletionCondition(DeletionCondition condition) {
		return deletionConditions.remove(condition);
	}
	
	public void clearDeletionConditions() {
		deletionConditions.clear();
	}

	public float getAlpha() {
		return alpha;
	}

	public void setAlpha(float alpha) {
		this.alpha = alpha;
	}

	public float getBeta() {
		return beta;
	}

	public void setBeta(float beta) {
		this.beta = beta;
	}

	public Vector3f getGravity() {
		return new Vector3f(gravity.x, gravity.y, gravity.z);
	}

	public void setGravity(Vector3f gravity) {
		this.gravity.x = gravity.x;
		this.gravity.y = gravity.y;
		this.gravity.z = gravity.z;
	}

	public float getLambda() {
		return lambda;
	}

	public void setLambda(float lambda) {
		this.lambda = lambda;
	}

	public float getMu() {
		return mu;
	}

	public void setMu(float mu) {
		this.mu = mu;
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
				
				String cacheID = DefaultAdvectionStrategy.class.getName();
				Program program = settings.getProgramCache().loadFromCache(cacheID);
				if (program == null) {
					program = settings.getClContext().createProgramFromSourceFiles(settings.getAssetManager(), SOURCE_FILE);
					program.build();
					settings.getProgramCache().saveToCache(cacheID, program);
				}
				program.register();
				kernels.DeletionBoxKernel = program.createKernel("DeletionBox").register();
				kernels.DeletionSphereKernel = program.createKernel("DeletionSphere").register();
				kernels.DeletionDensityThresholdKernel = program.createKernel("DeletionDensityThreshold").register();
				kernels.AdvectKernel = program.createKernel("Advect").register();
				
				kernelMap.put(settings, kernels);
			}
		}
	}

	@Override
	public void resized(int newSize) {
		//nothing to do
	}

	@Override
	public void advect(float tpf, int count, Buffer deletionBuffer) {

		Kernel.WorkSize ws = new Kernel.WorkSize(count);
		Buffer positionBuffer = controller.getBuffer(VertexBuffer.Type.Position).getCLBuffer();
		Buffer velocityBuffer = controller.getBuffer(VertexBuffer.Type.TexCoord2).getCLBuffer();
		Buffer temperatureBuffer = controller.getBuffer(VertexBuffer.Type.TexCoord3).getCLBuffer();
		
		//advect
		kernels.AdvectKernel.Run1NoEvent(kernels.clQueue, ws,
			positionBuffer, velocityBuffer, temperatureBuffer,
			alpha, beta, gravity, lambda, mu, tpf);
		
		//fill deletion buffer
		for (DeletionCondition c : deletionConditions) {
			if (c instanceof BoxDeletionCondition) {
				kernels.DeletionBoxKernel.Run1NoEvent(kernels.clQueue, ws,
						positionBuffer, deletionBuffer,
						((BoxDeletionCondition) c).min, ((BoxDeletionCondition) c).max);
			} else if (c instanceof SphereDeletionCondition) {
				kernels.DeletionSphereKernel.Run1NoEvent(kernels.clQueue, ws,
						positionBuffer, deletionBuffer,
						((SphereDeletionCondition) c).centerAndRadius);
			} else if (c instanceof DensityThresholdDeletionCondition) {
				kernels.DeletionDensityThresholdKernel.Run1NoEvent(kernels.clQueue, ws,
						positionBuffer, deletionBuffer,
						((DensityThresholdDeletionCondition) c).densityThreshold);
			}
		}

	}
	
	public static interface DeletionCondition {
		//marker interface
	}
	public static class BoxDeletionCondition implements DeletionCondition {
		private final Vector4f min;
		private final Vector4f max;

		public BoxDeletionCondition(Vector3f min, Vector3f max) {
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
			return "BoxDeletionCondition{" + "min=" + getMin() + ", max=" + getMax() + '}';
		}
	}
	public static class SphereDeletionCondition implements DeletionCondition {
		private final Vector4f centerAndRadius;
		
		public SphereDeletionCondition(Vector3f center, float radius) {
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
			return "SphereDeletionCondition{" + "center=" + getCenter() + ", radius=" + getRadius() + '}';
		}
	}
	public static class DensityThresholdDeletionCondition implements DeletionCondition {
		private final float densityThreshold;

		/**
		 * A particle is deleted when its density falls below the specified threshold
		 * @param densityThreshold the threshold on the density
		 */
		public DensityThresholdDeletionCondition(float densityThreshold) {
			this.densityThreshold = densityThreshold;
		}

		public float getDensityThreshold() {
			return densityThreshold;
		}

		@Override
		public String toString() {
			return "DensityThresholdDeletionCondition{" + "densityThreshold=" + densityThreshold + '}';
		}

	}
	
	private static class Kernels {
		private Kernel DeletionBoxKernel;
		private Kernel DeletionSphereKernel;
		private Kernel DeletionDensityThresholdKernel;
		private Kernel AdvectKernel;
		private CommandQueue clQueue;
	}
}

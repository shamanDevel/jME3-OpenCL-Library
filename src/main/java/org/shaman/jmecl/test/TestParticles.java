/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.test;

import com.jme3.app.SimpleApplication;
import com.jme3.material.Material;
import com.jme3.material.RenderState;
import com.jme3.math.Vector3f;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.renderer.queue.RenderQueue;
import com.jme3.system.AppSettings;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.particles.DefaultAdvectionStrategy;
import org.shaman.jmecl.particles.ParticleController;
import org.shaman.jmecl.particles.ShapeSeedingStrategy;
import org.shaman.jmecl.particles.ParticleRenderer;

/**
 *
 * @author Sebastian Weiss
 */
public class TestParticles extends SimpleApplication {

	private OpenCLSettings clSettings;
	private ParticleController particleController;
	private ParticleRenderer particleRenderer;
	
	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) {
		AppSettings settings = new AppSettings(true);
		settings.setOpenCLSupport(true);
		settings.setOpenCLPlatformChooser(UserPlatformChooser.class);
		settings.setVSync(true);
		TestParticles app = new TestParticles();
		app.setSettings(settings);
		app.setShowSettings(true);
		app.start();
	}

	@Override
	public void simpleInitApp() {
		Context clContext = context.getOpenCLContext();
		CommandQueue clCommandQueue = clContext.createQueue();
		clSettings = new OpenCLSettings(clContext, clCommandQueue, null, assetManager);
		
		ShapeSeedingStrategy seedingStrategy = new ShapeSeedingStrategy();
		//seedingStrategy.setShape(new ShapeSeedingStrategy.Sphere(Vector3f.ZERO, 1));
		seedingStrategy.setShape(new ShapeSeedingStrategy.Point(Vector3f.ZERO));
		seedingStrategy.setInitialVelocity(new Vector3f(0, 0.5f, 0));
		seedingStrategy.setVelocityVariation(0.1f);
		seedingStrategy.setInitialDensity(1f);
		seedingStrategy.setDensityVariation(0.1f);
		seedingStrategy.setInitialTemperature(0.9f);
		seedingStrategy.setTemperatureVariation(0.1f);
		seedingStrategy.setParticlesPerSecond(1000);
		
		DefaultAdvectionStrategy advectionStrategy = new DefaultAdvectionStrategy();
		advectionStrategy.setGravity(new Vector3f(0, -2f, 0));
		advectionStrategy.setAlpha(0.2f);
		advectionStrategy.setBeta(0.5f);
		advectionStrategy.setLambda(0.2f);
		advectionStrategy.setMu(0.1f);
		//advectionStrategy.addDeletionCondition(new DefaultAdvectionStrategy.BoxDeletionCondition(new Vector3f(-0.4f, -0.2f, -0.4f), new Vector3f(0.4f, 5f, 0.4f)));
		advectionStrategy.addDeletionCondition(new DefaultAdvectionStrategy.DensityThresholdDeletionCondition(0.1f));
		
		particleRenderer = new ParticleRenderer();
		particleController = new ParticleController(particleRenderer);
		particleController.initDefaultBuffers(1<<14);
		particleController.setSeedingStrategy(seedingStrategy);
		particleController.setAdvectionStrategy(advectionStrategy);
		particleController.init(renderManager, clSettings);
		
		Material mat = new Material(assetManager, "org/shaman/jmecl/particles/Particle.j3md");
        mat.setTexture("Texture", assetManager.loadTexture("org/shaman/jmecl/test/singlesmoke.png"));
		mat.setTexture("ColorRamp", assetManager.loadTexture("org/shaman/jmecl/test/ColorRamp.png"));
		mat.setFloat("SizeMultiplier", 500.0f);
		mat.setFloat("Alpha", 0.1f);
		mat.getAdditionalRenderState().setBlendMode(RenderState.BlendMode.AlphaAdditive);
        particleRenderer.setMaterial(mat);
		particleRenderer.setQueueBucket(RenderQueue.Bucket.Transparent);
		
		particleRenderer.addControl(particleController);
		rootNode.attachChild(particleRenderer);
	}
	
}

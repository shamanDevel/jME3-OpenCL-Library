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
import com.jme3.scene.Geometry;
import com.jme3.scene.shape.Quad;
import com.jme3.system.AppSettings;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.eq.EquationSolver;
import org.shaman.jmecl.fluids.*;
import org.shaman.jmecl.utils.SharedTexture;

/**
 *
 * @author Sebastian Weiss
 */
public class TestFluids2D extends SimpleApplication {

	private FluidSolver solver;
	private FlagGrid flags;
	private FlagGrid boundaryFlags;
	private RealGrid density;
	private MACGrid velocity;
	private SharedTexture densityTexture;
	
	private DebugTools debugTools;
	private BoundaryTools boundaryTools;
	private Buoyancy buoyancy;
	private Vector3f gravity;
	private Advection advection;
	private PressureProjection pressureProjection;
	
	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) {
		AppSettings settings = new AppSettings(true);
		settings.setOpenCLSupport(true);
		settings.setOpenCLPlatformChooser(UserPlatformChooser.class);
		settings.setVSync(true);
		TestFluids2D app = new TestFluids2D();
		app.setSettings(settings);
		app.setShowSettings(true);
		app.start();
	}

	@Override
	public void simpleInitApp() {
		Context clContext = context.getOpenCLContext();
		CommandQueue clCommandQueue = clContext.createQueue();
		OpenCLSettings clSettings = new OpenCLSettings(clContext, clCommandQueue, null, assetManager);
		
		int resolutionX = 256;
		int resolutionY = 256;
		solver = new FluidSolver(clSettings, resolutionX, resolutionY);
		flags = solver.createFlagGrid();
		flags.fill(FlagGrid.CellType.TypeFluid);
		boundaryFlags = solver.createFlagGrid();
		boundaryFlags.fill(FlagGrid.CellType.TypeEmpty);
		density = solver.createRealGrid();
		density.fill(0);
		velocity = solver.createMACGrid();
		velocity.fill(Vector3f.ZERO);
//		velocity.fill(new Vector3f(0.6f, 1.5f, 0));
		
		debugTools = new DebugTools(solver);
		densityTexture = debugTools.createDensityTexture(renderManager);
		Geometry densityTextureGeom = new Geometry("density", new Quad(resolutionX, resolutionY));
		Material densityTextureMat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
		densityTextureMat.setTexture("ColorMap", densityTexture.getJMETexture());
		//densityTextureMat.setTexture("ColorMap", assetManager.loadTexture("org/shaman/jmecl/test/singlesmoke.png"));
		densityTextureMat.getAdditionalRenderState().setFaceCullMode(RenderState.FaceCullMode.Off);
		densityTextureMat.getAdditionalRenderState().setBlendMode(RenderState.BlendMode.Alpha);
		densityTextureGeom.setMaterial(densityTextureMat);
		densityTextureGeom.setLocalTranslation(150, 150, 0);
		densityTextureGeom.setQueueBucket(RenderQueue.Bucket.Gui);
		guiNode.attachChild(densityTextureGeom);
		
		boundaryTools = new BoundaryTools(solver);
		boundaryTools.setFlagsInRect(boundaryFlags, FlagGrid.CellType.TypeInflow, new int[]{resolutionX/2-resolutionX/8, resolutionY/16}, new int[]{resolutionX/4, resolutionY/8});
		
		buoyancy = new Buoyancy(solver);
		gravity = new Vector3f(0, -9.81f, 0);
		advection = new Advection(solver);
		
		pressureProjection = new PressureProjection(solver);
		pressureProjection.setBoundary(flags);
		pressureProjection.setMaxError(EquationSolver.ERROR_DONT_TEST);
		pressureProjection.setMaxIterations(20000);
	}

	@Override
	public void simpleUpdate(float tpf) {
		float timestep = 0.1f;
		boundaryTools.applyDirichlet(density, boundaryFlags, FlagGrid.CellType.TypeInflow, 1.0f);

		advection.advect(velocity, density, timestep, 1);
		advection.advect(velocity, velocity, timestep, 1);
		
		buoyancy.addBuoynacy(flags, density, velocity, gravity, timestep);
		
//		System.out.println("Pre-Pressure");
//		debugTools.printGrid2D(velocity);
		pressureProjection.project(velocity);
//		System.out.println("Post-Pressure");
//		debugTools.printGrid2D(velocity);
		
		debugTools.fillTextureWithDensity2D(density, densityTexture);
		
//		try {
//			Thread.sleep(100);
//		} catch (InterruptedException ex) {
//			Logger.getLogger(TestFluids2D.class.getName()).log(Level.SEVERE, null, ex);
//		}
////		stop();
//		System.out.println();
//		System.out.println();
	}

}

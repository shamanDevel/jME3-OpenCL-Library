/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.test;

import com.jme3.app.SimpleApplication;
import com.jme3.app.StatsAppState;
import com.jme3.app.state.VideoRecorderAppState;
import com.jme3.material.Material;
import com.jme3.material.RenderState;
import com.jme3.math.Vector3f;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.renderer.queue.RenderQueue;
import com.jme3.scene.Geometry;
import com.jme3.scene.shape.Quad;
import com.jme3.system.AppSettings;
import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.eq.EquationSolver;
import org.shaman.jmecl.fluids.*;
import org.shaman.jmecl.utils.DebugContextFactory;
import org.shaman.jmecl.utils.LoggingContextFactory;
import org.shaman.jmecl.utils.SharedTexture;

/**
 *
 * @author Sebastian Weiss
 */
public class TestFluids2D_old extends SimpleApplication {
	private static final boolean RECORDING = false;
	private static final String RECORDING_PATH = "video/";
	private static final boolean DEBUG_CONTEXT = false;
	private static final boolean LOGGING_CONTEXT = false;

	private Context clContext;
	private CommandQueue clCommandQueue;
	private OpenCLSettings clSettings;
	
	private FluidSolver solver;
	private FlagGrid flags;
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
		TestFluids2D_old app = new TestFluids2D_old();
		app.setSettings(settings);
		app.setShowSettings(true);
		app.start();
	}

	@Override
	public void simpleInitApp() {
		clContext = context.getOpenCLContext();
		if (DEBUG_CONTEXT) {
			clContext = DebugContextFactory.createDebugContext(clContext);
		}
		if (LOGGING_CONTEXT) {
			clContext = LoggingContextFactory.createLoggingContext(clContext);
		}
		clCommandQueue = clContext.createQueue();
		clSettings = new OpenCLSettings(clContext, clCommandQueue, null, assetManager);
		
		int resolutionX = 512;
		int resolutionY = 512;
		solver = new FluidSolver(clSettings, resolutionX, resolutionY);
		flags = solver.createFlagGrid();
		flags.fill(FlagGrid.CellType.TypeFluid);
		density = solver.createRealGrid();
		density.fill(0);
		velocity = solver.createMACGrid();
		velocity.fill(Vector3f.ZERO);
//		velocity.fill(new Vector3f(0.6f, 1.5f, 0));
		
		debugTools = new DebugTools(solver);
		densityTexture = debugTools.createRealTexture2D(renderManager);
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
		boundaryTools.setFlagsInRect(flags, FlagGrid.CellType.TypeObstacle.value, new int[]{0,0}, new int[]{2, resolutionY});
		boundaryTools.setFlagsInRect(flags, FlagGrid.CellType.TypeObstacle.value, new int[]{resolutionX-2,0}, new int[]{2, resolutionY});
		boundaryTools.setFlagsInRect(flags, FlagGrid.CellType.TypeObstacle.value, new int[]{0,resolutionY-2}, new int[]{resolutionX, 2});
		boundaryTools.setFlagsInRect(flags, FlagGrid.CellType.TypeOutflow.value, new int[]{0,0}, new int[]{resolutionX, 2});
		boundaryTools.setFlagsInRect(flags, FlagGrid.CellType.TypeInflow.value | FlagGrid.CellType.TypeFluid.value,
				new int[]{resolutionX/2-resolutionX/8, resolutionY/16}, new int[]{resolutionX/4, resolutionY/8});
		boundaryTools.setFlagsInRect(flags, FlagGrid.CellType.TypeObstacle.value, new int[]{resolutionX/2-resolutionX/8, resolutionY/2}, new int[]{resolutionX/4, resolutionY/8});
		
		
		buoyancy = new Buoyancy(solver);
		gravity = new Vector3f(0, -9.81f, 0);
		advection = new Advection(solver);
		
		pressureProjection = new PressureProjection(solver);
		pressureProjection.setBoundary(flags);
		pressureProjection.setMaxError(EquationSolver.ERROR_ONLY_TEST_AT_THE_END);
		pressureProjection.setMaxIterations(2000);
		
		if (RECORDING) {
			File folder = new File(RECORDING_PATH);
			if (!folder.exists()) {
				folder.mkdir();
			}
			File file;
			for (int i=1; ;++i) {
				file = new File(folder, "Video"+i+".avi");
				if (!file.exists()) {
					break;
				}
			}
			VideoRecorderAppState vras = new VideoRecorderAppState(file, 1.0f, 30);
			stateManager.attach(vras);
		}
	}

	@Override
	public void simpleUpdate(float tpf) {
		if (RECORDING && stateManager.getState(StatsAppState.class) != null) {
			stateManager.getState(StatsAppState.class).setEnabled(false);
		}
		
		float timestep = RECORDING ? tpf : 0.1f;
		boundaryTools.applyDirichlet(density, flags, FlagGrid.CellType.TypeInflow, 1.0f);

		advection.advect(velocity, density, timestep, 1);
		advection.advect(velocity, velocity, timestep, 1);
		
		buoyancy.addBuoynacy(flags, density, velocity, gravity, timestep);
		
//		System.out.println("Pre-Pressure");
//		debugTools.printGrid2D(velocity);
		pressureProjection.project(velocity);
//		System.out.println("Post-Pressure");
//		debugTools.printGrid2D(velocity);
//		pressureProjection.debugPrintDivergence(velocity);
		
		clCommandQueue.finish();
		debugTools.fillTextureWithDensity2D(density, densityTexture);
		clCommandQueue.finish();
		
//		try {
//			Thread.sleep(100);
//		} catch (InterruptedException ex) {
//			Logger.getLogger(TestFluids2D.class.getName()).log(Level.SEVERE, null, ex);
//		}
//		stop();
//		System.out.println();
//		System.out.println();

		if (RECORDING) {
			System.out.println(".");
		}
	}

}

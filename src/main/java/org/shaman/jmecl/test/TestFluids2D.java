/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.test;

import com.jme3.app.SimpleApplication;
import com.jme3.math.Vector3f;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.system.AppSettings;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.fluids.Buoyancy;
import org.shaman.jmecl.fluids.FlagGrid;
import org.shaman.jmecl.fluids.FluidSolver;
import org.shaman.jmecl.fluids.MACGrid;
import org.shaman.jmecl.fluids.RealGrid;

/**
 *
 * @author Sebastian Weiss
 */
public class TestFluids2D extends SimpleApplication {

	private FluidSolver solver;
	private FlagGrid flags;
	private RealGrid density;
	private MACGrid velocity;
	
	private Buoyancy buoyancy;
	private Vector3f gravity;
	
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
		
		int resolutionX = 128;
		int resolutionY = 128;
		solver = new FluidSolver(clSettings, resolutionX, resolutionY);
		flags = solver.createFlagGrid();
		flags.fill(FlagGrid.CellType.TypeFluid);
		density = solver.createRealGrid();
		density.fill(0);
		velocity = solver.createMACGrid();
		velocity.fill(Vector3f.ZERO);
		
		buoyancy = new Buoyancy(solver);
		gravity = new Vector3f(0, -9.81f, 0);
	}

	@Override
	public void simpleUpdate(float tpf) {
	
		float timestep = tpf;
		buoyancy.addBuoynacy(flags, density, velocity, gravity, timestep);
		
	}

}

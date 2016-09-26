/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.test;

import com.jme3.math.Vector3f;
import com.jme3.system.AppSettings;
import org.shaman.jmecl.eq.EquationSolver;
import org.shaman.jmecl.fluids.Advection;
import org.shaman.jmecl.fluids.BoundaryTools;
import org.shaman.jmecl.fluids.Buoyancy;
import org.shaman.jmecl.fluids.FlagGrid;
import org.shaman.jmecl.fluids.FluidSolver;
import org.shaman.jmecl.fluids.MACGrid;
import org.shaman.jmecl.fluids.PressureProjection;
import org.shaman.jmecl.fluids.RealGrid;

/**
 *
 * @author Sebastian
 */
public class TestFluids2D extends AbstractFluidTest2D {
	
	private FlagGrid flags;
	private RealGrid density;
	private MACGrid velocity;
	
	private BoundaryTools boundaryTools;
	private Buoyancy buoyancy;
	private Vector3f gravity;
	private Advection advection;
	private PressureProjection pressureProjection;
	
	public static void main(String[] args) {
		AppSettings settings = new AppSettings(true);
		settings.setOpenCLSupport(true);
		settings.setOpenCLPlatformChooser(UserPlatformChooser.class);
		settings.setVSync(false);
		settings.setResolution(1024, 768);
		TestFluids2D app = new TestFluids2D();
		app.setSettings(settings);
		app.setShowSettings(true);
		app.start();
	}

	public TestFluids2D() {
		setResolutionX(512);
		setResolutionY(512);
	}

	@Override
	protected void initSolver(FluidSolver solver) {
		flags = solver.createFlagGrid();
		flags.fill(FlagGrid.CellType.TypeFluid);
		density = solver.createRealGrid();
		density.fill(0);
		velocity = solver.createMACGrid();
		velocity.fill(Vector3f.ZERO);
		
		setSelectedRealGrid(addRealGrid(density, "Density"));
		setSelectedFlagGrid(addFlagGrid(flags, "Flags"));
		setSelectedMACGrid(addMACGrid(velocity, "Velocity"));
		
		int resolutionX = solver.getResolutionX();
		int resolutionY = solver.getResolutionY();
		
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
	}

	@Override
	protected void updateSolver(float tpf) {
		float timestep = tpf;
		boundaryTools.applyDirichlet(density, flags, FlagGrid.CellType.TypeInflow, 1.0f);

		advection.advect(velocity, density, timestep, 1);
		advection.advect(velocity, velocity, timestep, 1);
		
		buoyancy.addBuoynacy(flags, density, velocity, gravity, timestep);

		pressureProjection.project(velocity);
	}
	
}

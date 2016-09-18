/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.MappingAccess;
import com.jme3.opencl.MemoryAccess;
import com.jme3.renderer.RenderManager;
import com.jme3.texture.Image;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.shaman.jmecl.utils.SharedTexture;

/**
 * Various helper methods for testing the fluid solver.
 * @author Sebastian
 */
public class DebugTools {
	
	private final FluidSolver solver;

	public DebugTools(FluidSolver solver) {
		this.solver = solver;
	}
	
	public SharedTexture createDensityTexture(RenderManager renderManager) {
		com.jme3.opencl.Image.ImageDescriptor descriptor = new com.jme3.opencl.Image.ImageDescriptor(
				com.jme3.opencl.Image.ImageType.IMAGE_2D, solver.resolutionX, solver.resolutionY, 0, 0);
		SharedTexture tex = new SharedTexture(descriptor, Image.Format.Luminance32F);
		tex.initialize(renderManager, solver.clSettings.getClContext(), MemoryAccess.READ_WRITE);
		return tex;
	}
	
	public void fillTextureWithDensity2D(RealGrid density, SharedTexture texture) {
		CommandQueue cq = solver.clSettings.getClCommandQueue();
		texture.getCLImage().acquireImageForSharingNoEvent(cq);
		density.buffer.copyToImageAsync(cq, texture.getCLImage(), 0, new long[]{0,0,0}, new long[]{solver.resolutionX, solver.resolutionY,1}).release();
		texture.getCLImage().releaseImageForSharingNoEvent(cq);
	}
	
	public void printGrid2D(RealGrid grid) {
		ByteBuffer bb = grid.buffer.map(solver.clSettings.getClCommandQueue(), MappingAccess.MAP_READ_ONLY);
		FloatBuffer fb = bb.asFloatBuffer();
		for (int y=0; y<solver.resolutionY; ++y) {
			for (int x=0; x<solver.resolutionX; ++x) {
				float v = fb.get();
				System.out.printf("%2.3f ", v);
			}
			System.out.println();
		}
		grid.buffer.unmap(solver.clSettings.getClCommandQueue(), bb);
	}
	
	public void printGrid2D(MACGrid grid) {
		ByteBuffer bb = grid.buffer.map(solver.clSettings.getClCommandQueue(), MappingAccess.MAP_READ_ONLY);
		FloatBuffer fb = bb.asFloatBuffer();
		for (int y=0; y<solver.resolutionY; ++y) {
			for (int x=0; x<solver.resolutionX; ++x) {
				int idx1 = x + y*(solver.resolutionX+1);
				int idx2 = (solver.resolutionX+1)*solver.resolutionY + x + y*solver.resolutionX;
				float vx = 0.5f * (fb.get(idx1) + fb.get(idx1+1));
				float vy = 0.5f * (fb.get(idx2) + fb.get(idx2+solver.resolutionX));
				System.out.printf("(%2.3f,%2.3f) ", vx, vy);
			}
			System.out.println();
		}
		grid.buffer.unmap(solver.clSettings.getClCommandQueue(), bb);
	}
}

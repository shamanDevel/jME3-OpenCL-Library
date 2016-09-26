/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.fluids;

import com.jme3.math.ColorRGBA;
import com.jme3.opencl.Buffer;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.MappingAccess;
import com.jme3.opencl.MemoryAccess;
import com.jme3.opencl.Program;
import com.jme3.renderer.RenderManager;
import com.jme3.texture.Image;
import com.jme3.util.BufferUtils;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.EnumMap;
import org.shaman.jmecl.utils.SharedTexture;

/**
 * Various helper methods for testing the fluid solver.
 * @author Sebastian
 */
public class DebugTools {
	
	private final FluidSolver solver;
	
	private boolean flagCopyInitialized;
	private Buffer flagColorsBuffer;
	private Kernel flagCopyKernel;

	public DebugTools(FluidSolver solver) {
		this.solver = solver;
	}
	
	public SharedTexture createRealTexture2D(RenderManager renderManager) {
		com.jme3.opencl.Image.ImageDescriptor descriptor = new com.jme3.opencl.Image.ImageDescriptor(
				com.jme3.opencl.Image.ImageType.IMAGE_2D, solver.resolutionX, solver.resolutionY, 0, 0);
		SharedTexture tex = new SharedTexture(descriptor, Image.Format.Luminance32F);
		tex.initialize(renderManager, solver.clSettings.getClContext(), MemoryAccess.READ_WRITE);
		return tex;
	}
	
	public void fillTextureWithDensity2D(RealGrid density, SharedTexture realTexture) {
		CommandQueue cq = solver.clSettings.getClCommandQueue();
		realTexture.getCLImage().acquireImageForSharingNoEvent(cq);
		density.buffer.copyToImageAsync(cq, realTexture.getCLImage(), 0, new long[]{0,0,0}, new long[]{solver.resolutionX, solver.resolutionY,1}).release();
		realTexture.getCLImage().releaseImageForSharingNoEvent(cq);
	}
	
	public SharedTexture createFlagTexture2D(RenderManager renderManager) {
		com.jme3.opencl.Image.ImageDescriptor descriptor = new com.jme3.opencl.Image.ImageDescriptor(
				com.jme3.opencl.Image.ImageType.IMAGE_2D, solver.resolutionX, solver.resolutionY, 0, 0);
		SharedTexture tex = new SharedTexture(descriptor, Image.Format.RGBA8);
		tex.initialize(renderManager, solver.clSettings.getClContext(), MemoryAccess.READ_WRITE);
		return tex;
	}
	
	private void initFlagCopy() {
		if (flagCopyInitialized) {
			return;
		}
		
		String cacheID = DebugTools.class.getName()+"_FlagCopy";
		Program program = solver.clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			String source = 
				"#define L_INVSZ(i, vi, sz)	(vi).y = i / (sz).x; (vi).x = (i - (vi).y*(sz).x);\n" +
				"__kernel void CopyFlags(__global char* flagGrid, __global float4* colors, __write_only image2d_t image, int sizeX, int sizeY)\n" +
				"{\n" +
				"	int idx = get_global_id(0);\n" +
				"	int2 dim = (int2)(sizeX, sizeY);\n" +
				"	int2 pos;\n" +
				"	L_INVSZ(idx, pos, dim);\n" +
				"	char flag = flagGrid[idx];\n" +
				"	float4 color = colors[flag];\n" +
				"	write_imagef(image, pos, color);\n" +
				"}";
			program = solver.clSettings.getClContext().createProgramFromSourceCode(source);
			program.build();
			solver.clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		flagCopyKernel = program.createKernel("CopyFlags");
		
		ByteBuffer bb = BufferUtils.createByteBuffer(256*4*4);
		FloatBuffer fb = bb.asFloatBuffer();
		EnumMap<FlagGrid.CellType, ColorRGBA> flagColors = new EnumMap<>(FlagGrid.CellType.class);
		flagColors.put(FlagGrid.CellType.TypeInflow, new ColorRGBA(0, 1, 0, 0.5f));
		flagColors.put(FlagGrid.CellType.TypeOutflow, new ColorRGBA(0, 0, 1, 0.5f));
		flagColors.put(FlagGrid.CellType.TypeObstacle, new ColorRGBA(1, 0, 0, 1));
		for (int i=0; i<256; ++i) {
			ColorRGBA col = new ColorRGBA(0, 0, 0, 0);
			for (EnumMap.Entry<FlagGrid.CellType, ColorRGBA> e : flagColors.entrySet()) {
				if ((i & e.getKey().value) != 0) {
					ColorRGBA c = e.getValue();
					float ac = c.a + (1-c.a)*col.a;
					col.set((c.a*c.r + (1-c.a)*col.a*col.r)/ac, (c.a*c.g + (1-c.a)*col.a*col.g)/ac, (c.a*c.b + (1-c.a)*col.a*col.b)/ac, ac);
				}
			}
			System.out.println("flag "+i+" ("+Integer.toBinaryString(i)+"): "+col);
			fb.put(col.r).put(col.g).put(col.b).put(col.a);
		}
		flagColorsBuffer = solver.clSettings.getClContext().createBuffer(256*4*4);
		flagColorsBuffer.write(solver.clSettings.getClCommandQueue(), bb);
		
		flagCopyInitialized = true;
	}
	
	public void fillTextureWithFlags2D(FlagGrid flags, SharedTexture flagTexture) {
		initFlagCopy();
		CommandQueue cq = solver.clSettings.getClCommandQueue();
		flagTexture.getCLImage().acquireImageForSharingNoEvent(cq);
		Kernel.WorkSize ws = new Kernel.WorkSize(solver.resolutionX * solver.resolutionY);
		flagCopyKernel.Run1NoEvent(cq, ws, flags.buffer, flagColorsBuffer, flagTexture.getCLImage(), solver.resolutionX, solver.resolutionY);
		flagTexture.getCLImage().releaseImageForSharingNoEvent(cq);
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

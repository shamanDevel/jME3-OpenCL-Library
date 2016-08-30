/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.eq;

import com.jme3.opencl.Buffer;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.MappingAccess;
import com.jme3.opencl.Program;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.utils.CLBlas;

/**
 * A very simple jacobi solver.
 * @author Sebastian
 */
public class JacobiSolver extends EquationSolver {

	private static final Logger LOG = Logger.getLogger(JacobiSolver.class.getName());
	private static final String SOURCE_FILE = "org/shaman/jmecl/eq/JacobiSolver.cl";
	
	protected Buffer bufXCopy;
	protected Buffer bufA;
	protected Buffer bufRes;
	protected Kernel iteration2DKernel;
	protected CLBlas<Float> blas;
	
	public JacobiSolver(OpenCLSettings clSettings, int resolutionX, int resolutionY, int resolutionZ) {
		super(clSettings, resolutionX, resolutionY, resolutionZ);
		long size = 4 * resolutionX * resolutionY * resolutionZ;
		this.bufXCopy = clSettings.getClContext().createBuffer(size);
	}

	public JacobiSolver(OpenCLSettings clSettings, int resolutionX, int resolutionY) {
		super(clSettings, resolutionX, resolutionY);
		long size = 4 * resolutionX * resolutionY * resolutionZ;
		this.bufXCopy = clSettings.getClContext().createBuffer(size);
		this.bufRes = clSettings.getClContext().createBuffer(size);
		this.bufA = clSettings.getClContext().createBuffer(5 * size);
		
		String cacheID = JacobiSolver.class.getName();
		Program program = clSettings.getProgramCache().loadFromCache(cacheID);
		if (program == null) {
			program = clSettings.getClContext().createProgramFromSourceFiles(clSettings.getAssetManager(), SOURCE_FILE);
			program.build();
			clSettings.getProgramCache().saveToCache(cacheID, program);
		}
		program.register();
		iteration2DKernel = program.createKernel("Iteration2D").register();
		blas = CLBlas.get(clSettings, Float.class);
	}
	
	@Override
	public void setA(Buffer buf, int stencilX, int stencilY, int stencilZ) {
		checkStencil(buf, stencilX, stencilY, stencilZ);
		if (is2D()) {
			long offset = 0;
			if (stencilX==-1) {
				offset = 1;
			} else if (stencilX==1) {
				offset = 2;
			} else if (stencilY==-1) {
				offset = 3;
			} else if (stencilY==1) {
				offset = 4;
			}
			long size = 4 * resolutionX * resolutionY;
			offset *= size;
			buf.copyTo(clSettings.getClCommandQueue(), bufA, size, 0, offset);
		}
	}

	@Override
	public void solve(int maxIteration, float maxError) {
		CommandQueue q = clSettings.getClCommandQueue();
		if (is2D()) {
			int i;
			float residuum = 0;
			Buffer bufs[] = new Buffer[]{bufX, bufXCopy};
			Kernel.WorkSize ws = new Kernel.WorkSize(resolutionX * resolutionY);
			CLBlas.ReduceResult reduceResult = new CLBlas.ReduceResult();
			for (i=0; i<maxIteration; ++i) {
				int b1 = i%2;
				int b2 = (i+1)%2;
				iteration2DKernel.Run1NoEvent(q, ws, bufs[b1], bufs[b2], bufB, bufA, resolutionX, resolutionY, bufRes);
				
				/*
				System.out.println("x:");
				ByteBuffer bb = bufs[b2].map(q, MappingAccess.MAP_WRITE_ONLY);
				FloatBuffer fb = bb.asFloatBuffer();
				for (int y=0; y<resolutionY; ++y) {
					for (int x=0; x<resolutionX; ++x) {
						float v = fb.get();
						System.out.printf("%2.2f ", v);
					}
					System.out.println();
				}
				bufs[b2].unmap(q, bb);
				System.out.println();
				*/
				
				reduceResult = blas.reduce(bufRes, CLBlas.PreReduceOp.SQUARE, CLBlas.ReduceOp.ADD, reduceResult);
				residuum = blas.getReduceResultBlocking(reduceResult);
				if (residuum < maxError) {
					i++;
					break;
				}
				LOG.info("Iteration "+i+": residium="+residuum);
			}
			if (i%2 == 1) {
				bufs[1].copyToAsync(q, bufs[0]).release();
			}
			LOG.log(Level.INFO, "solved after {0} iterations with an error of {1}", new Object[]{i, residuum});
		}
	}
	
}

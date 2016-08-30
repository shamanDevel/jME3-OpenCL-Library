/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl;

import com.jme3.opencl.Buffer;
import com.jme3.opencl.MappingAccess;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.shaman.jmecl.eq.JacobiSolver;
import org.shaman.jmecl.utils.CLBlas;

/**
 *
 * @author Sebastian
 */
public class SolverTest extends AbstractOpenCLTest {
	
	public SolverTest() {
	}

	@Test
	public void testJacobi2D() {
		int resX = 32;
		int resY = 32;
		float beta = 0.6f;
		JacobiSolver solver = new JacobiSolver(settings, resX, resY);
		//build matrix
		Buffer buf = settings.getClContext().createBuffer(resX * resY * 4);
		CLBlas<Float> blas = CLBlas.get(settings, Float.class);
		blas.fill(buf, 1 + 4*beta);
		solver.setA(buf, 0, 0, 0);
		blas.fill(buf, -beta);
		solver.setA(buf, -1, 0, 0);
		solver.setA(buf, 1, 0, 0);
		solver.setA(buf, 0, -1, 0);
		solver.setA(buf, 0, 1, 0);
		solver.assembleMatrix();
		ByteBuffer bb = solver.getBBuffer().map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
		FloatBuffer fb = bb.asFloatBuffer();
		for (int y=0; y<resY; ++y) {
			for (int x=0; x<resX; ++x) {
				if (x>=resX/4 && x<resX*3/4 && y>=resY/4 && y<resY*3/4) {
					fb.put(1);
				} else {
					fb.put(0);
				}
			}
		}
		solver.getBBuffer().unmap(clCommandQueue, bb);
		//solve
		solver.setXToZero();
		solver.solve(50, 1e-5f);
		//check result
		bb = solver.getXBuffer().map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		fb = bb.asFloatBuffer();
		for (int y=0; y<resY; ++y) {
			for (int x=0; x<resX; ++x) {
				float v = fb.get();
				System.out.printf("%2.2f ", v);
			}
			System.out.println();
		}
		solver.getXBuffer().unmap(clCommandQueue, bb);
	}
}

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
import java.util.Arrays;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;
import org.la4j.vector.dense.BasicVector;
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
		
		Matrix matA = new Basic2DMatrix(resX * resY, resX * resY);
		for (int x=0; x<resX; ++x) {
			for (int y=0; y<resY; ++y) {
				int row = x + resX * y;
				matA.set(row, row, 1 + 4*beta);
				if (x>0) matA.set(row, (x-1) + resX*y, -beta);
				if (x<resX-1) matA.set(row, (x+1) + resX*y, -beta);
				if (y>0) matA.set(row, x + resX*(y-1), -beta);
				if (y<resY-1) matA.set(row, x + resX*(y+1), -beta);
			}
		}
		
		//build rhs
		ByteBuffer bb = solver.getBBuffer().map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
		FloatBuffer fb = bb.asFloatBuffer();
		Vector vecB = new BasicVector(resX * resY);
		for (int y=0; y<resY; ++y) {
			for (int x=0; x<resX; ++x) {
				float v;
				if (x>=resX/4 && x<resX*3/4 && y>=resY/4 && y<resY*3/4) {
					v = 1;
				} else {
					v = 0;
				}
				fb.put(v);
				vecB.set(x + resX * y, v);
			}
		}
		solver.getBBuffer().unmap(clCommandQueue, bb);
		
		//solve
		solver.setXToZero();
		solver.solve(50, 1e-5f);
		//get result
		bb = solver.getXBuffer().map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		fb = bb.asFloatBuffer();
		Vector vecX = new BasicVector(resX * resY);
		for (int i=0; i<resX*resY; ++i) {
			vecX.set(i, fb.get());
		}
		fb.rewind();
		
		//print
		for (int y=0; y<resY; ++y) {
			for (int x=0; x<resX; ++x) {
				float v = fb.get();
				System.out.printf("%2.2f ", v);
			}
			System.out.println();
		}
		solver.getXBuffer().unmap(clCommandQueue, bb);
		
		//check against cpu version
		org.la4j.linear.JacobiSolver la4jSolver = new org.la4j.linear.JacobiSolver(matA);
		Vector vecX2 = la4jSolver.solve(vecB);
		double difference = vecX.subtract(vecX2).norm();
		System.out.println("difference of cpu and gpu version: "+difference);
		assertTrue("CPU and GPU match", difference < 0.01);
	}
	
	@Test
	public void testJacobi3D() {
		int resX = 8;
		int resY = 8;
		int resZ = 8;
		float beta = 0.6f;
		JacobiSolver solver = new JacobiSolver(settings, resX, resY, resZ);
		
		//build matrix
		Buffer buf = settings.getClContext().createBuffer(resX * resY * resZ * 4);
		CLBlas<Float> blas = CLBlas.get(settings, Float.class);
		blas.fill(buf, 1 + 6*beta);
		solver.setA(buf, 0, 0, 0);
		blas.fill(buf, -beta);
		solver.setA(buf, -1, 0, 0);
		solver.setA(buf, 1, 0, 0);
		solver.setA(buf, 0, -1, 0);
		solver.setA(buf, 0, 1, 0);
		solver.setA(buf, 0, 0, -1);
		solver.setA(buf, 0, 0, 1);
		solver.assembleMatrix();
		
		//Matrix matA = new Basic2DMatrix(resX * resY * resZ, resX * resY * resZ);
		double[] matAValues = new double[resX * resY * resZ * 7];
		int[] matAColumns = new int[resX * resY * resZ * 7];
		int[] matARows = new int[resX * resY * resZ + 1];
		int matAI = 0;
		for (int z=0; z<resZ; ++z) {
			for (int y=0; y<resY; ++y) {
				for (int x=0; x<resX; ++x) {
					int row = x + resX * (y + resY * z);
					matARows[row] = matAI;
					matAColumns[matAI] = row; matAValues[matAI] = 1 + 6*beta; ++matAI;
					if (x>0) {
						matAColumns[matAI] = (x-1) + resX * (y + resY * z); matAValues[matAI] = -beta; ++matAI;
					}
					if (x<resX-1) {
						matAColumns[matAI] = (x+1) + resX * (y + resY * z); matAValues[matAI] = -beta; ++matAI;
					}
					if (y>0) {
						matAColumns[matAI] = x + resX * (y-1 + resY * z); matAValues[matAI] = -beta; ++matAI;
					}
					if (y<resY-1) {
						matAColumns[matAI] = x + resX * (y+1 + resY * z); matAValues[matAI] = -beta; ++matAI;
					}
					if (z>0) {
						matAColumns[matAI] = x + resX * (y + resY * (z-1)); matAValues[matAI] = -beta; ++matAI;
					}
					if (z<resZ-1) {
						matAColumns[matAI] = x + resX * (y + resY * (z+1)); matAValues[matAI] = -beta; ++matAI;
					}
				}
			}
		}
		matAColumns = Arrays.copyOf(matAColumns, matAI);
		matAValues = Arrays.copyOf(matAValues, matAI);
//		matARows[resX*resY*resZ] = matAI;
		Matrix matA = new CRSMatrix(resX * resY * resZ, resX * resY * resZ, matAI, matAValues, matAColumns, matARows);
		matA = matA.toDenseMatrix();
		System.out.println(matA);
		
		//build rhs
		ByteBuffer bb = solver.getBBuffer().map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
		FloatBuffer fb = bb.asFloatBuffer();
		Vector vecB = new BasicVector(resX * resY * resZ);
		for (int z=0; z<resZ; ++z) {
			for (int y=0; y<resY; ++y) {
				for (int x=0; x<resX; ++x) {
					float v;
					if (x>=resX/4 && x<resX*3/4 && y>=resY/4 && y<resY*3/4 && z>=resZ/4 && z<resZ*3/4) {
						v = 1;
					} else {
						v = 0;
					}
					fb.put(v);
					vecB.set(x + resX * (y + resY * z), v);
				}
			}
		}
		solver.getBBuffer().unmap(clCommandQueue, bb);
		
		//solve
		solver.setXToZero();
		solver.solve(50, 1e-5f);
		//get result
		bb = solver.getXBuffer().map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		fb = bb.asFloatBuffer();
		Vector vecX = new BasicVector(resX * resY * resZ);
		for (int i=0; i<resX*resY; ++i) {
			vecX.set(i, fb.get());
		}
		solver.getXBuffer().unmap(clCommandQueue, bb);
		
		//check against cpu version
		org.la4j.linear.LinearSystemSolver la4jSolver = new org.la4j.linear.SeidelSolver(matA);
		Vector vecX2 = la4jSolver.solve(vecB);
		double difference = vecX.subtract(vecX2).norm();
		System.out.println("difference of cpu and gpu version: "+difference);
		assertTrue("CPU and GPU match", difference < 0.01);
	}
}

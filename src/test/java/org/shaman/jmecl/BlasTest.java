/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl;

import com.jme3.asset.AssetManager;
import com.jme3.asset.DesktopAssetManager;
import com.jme3.opencl.*;
import java.nio.ByteBuffer;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.shaman.jmecl.utils.CLBlas;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author Sebastian Weiss
 */
public class BlasTest extends AbstractOpenCLTest {
	private static final Logger LOG = Logger.getLogger(BlasTest.class.getName());
	
	public BlasTest() {
	}
	
	@Test
	public void testFill() {
		Buffer b = clContext.createBuffer(4 * 16);
		CLBlas<Integer> blas = CLBlas.get(settings, Integer.class);
		blas.fill(b, 5).release();
		blas.fill(b, 2, 2).release();
		blas.fill(b, 7, 3, 4, 2).release();
		ByteBuffer buf = b.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		assertEquals(2, buf.getInt());
		assertEquals(2, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(7, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(7, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(7, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
		assertEquals(5, buf.getInt());
	}
	
	@Test
	public void testAXPY() {
		int s = 16;
		Buffer a = clContext.createBuffer(4 * s);
		Buffer b = clContext.createBuffer(4 * s);
		Buffer c = clContext.createBuffer(4 * s);
		
		ByteBuffer buf = a.map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
		for (int i=0; i<s; ++i) {
			buf.putFloat(i);
		}
		buf.rewind();
		a.unmap(clCommandQueue, buf);
		
		buf = b.map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
		for (int i=s; i>0; --i) {
			buf.putFloat(2*i);
		}
		buf.rewind();
		b.unmap(clCommandQueue, buf);
		
		CLBlas<Float> blas = CLBlas.get(settings, Float.class);
		blas.axpy(2.0f, a, b, c).release();
		
		buf = c.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		for (int i=0; i<s; ++i) {
			assertEquals(2.0f*s, buf.getFloat(), 0.000001f);
		}
		c.unmap(clCommandQueue, buf);
	}
	
	@Test
	public void testMap() {
		CLBlas<Double> blas = CLBlas.get(settings, Double.class);
		Buffer x = clContext.createBuffer(8 * 8);
		Buffer z = clContext.createBuffer(8 * 8);
		blas.fill(x, 0.0).release();
		
		//set
		blas.map(x, CLBlas.MapOp.SET, 5.0, z).release();
		assertBufferEquals(z, 0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
		//add
		blas.map(z, CLBlas.MapOp.ADD, 1.5, x, 3).release();
		assertBufferEquals(x, 0, 6.5, 6.5, 6.5, 0.0, 0.0, 0.0, 0.0, 0.0);
		//sub
		blas.map(z, CLBlas.MapOp.SUB, 7.0, x, 2, 3, 3, 2, 2).release();
		assertBufferEquals(x, 0, 6.5, 6.5, 6.5, -2.0, 0.0, -2.0, 0.0, 0.0);
	}
	
	@Test
	public void testReduce1() {
		CLBlas<Integer> blas = CLBlas.get(settings, Integer.class);
		
		int sizes[] = {4, 25, 246, 1<<12, 238561};
		CLBlas.ReduceResult result = new CLBlas.ReduceResult();
		for (int size : sizes) {
			Buffer x = clContext.createBuffer(size * 4);
			
			blas.fill(x, -2);
			result = blas.reduce(x, CLBlas.PreReduceOp.NONE, CLBlas.ReduceOp.ADD, result);
			assertEquals(-2 * size, (int) blas.getReduceResultBlocking(result));
			
			if (size < 30) {
				result = blas.reduce(x, CLBlas.PreReduceOp.NONE, CLBlas.ReduceOp.MUL, result);
				assertEquals((int) Math.round(Math.pow(-2, size)), (int) blas.getReduceResultBlocking(result));
			}
			
			result = blas.reduce(x, CLBlas.PreReduceOp.ABS, CLBlas.ReduceOp.ADD, size / 2, result);
			assertEquals(2 * (size / 2), (int) blas.getReduceResultBlocking(result));
			
			blas.fill(x, -3, 2);
			result = blas.reduce(x, CLBlas.PreReduceOp.SQUARE, CLBlas.ReduceOp.ADD, size/3, 0, 2, result);
			assertEquals(4 * (size/3) + 5, (int) blas.getReduceResultBlocking(result));
			
			ByteBuffer bbuf = x.map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
			int minValue = Integer.MAX_VALUE;
			int maxValue = Integer.MIN_VALUE;
			bbuf.rewind();
			for (int i=0; i<size; ++i) {
				int v = randInt(-100000, 100000);
				minValue = Math.min(v, minValue);
				maxValue = Math.max(v, maxValue);
				bbuf.putInt(v);
			}
			bbuf.rewind();
			x.unmap(clCommandQueue, bbuf);
			result = blas.reduce(x, CLBlas.PreReduceOp.NONE, CLBlas.ReduceOp.MIN, result);
			assertEquals(minValue, (int) blas.getReduceResultBlocking(result));
			result = blas.reduce(x, CLBlas.PreReduceOp.NONE, CLBlas.ReduceOp.MAX, result);
			assertEquals(maxValue, (int) blas.getReduceResultBlocking(result));
			
			x.release();
		}
	}
	
	@Test
	public void testReduce2() {
		CLBlas<Double> blas = CLBlas.get(settings, Double.class);
		
		int performanceSize = 238561;
		int performanceCount = 1000;
		int sizes[] = {4, 25, 246, 1<<12, performanceSize};
		CLBlas.ReduceResult result = new CLBlas.ReduceResult();
		for (int size : sizes) {
			Buffer a = clContext.createBuffer(size * 8);
			Buffer b = clContext.createBuffer(size * 8);
			
			//test dot-product
			ByteBuffer bufA = a.map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
			ByteBuffer bufB = b.map(clCommandQueue, MappingAccess.MAP_WRITE_ONLY);
			bufA.rewind();
			bufB.rewind();
			double dotP = 0;
			for (int i=0; i<size; ++i) {
				double va = randDouble(-1, 1);
				double vb = randDouble(-1, 1);
				dotP += va * vb;
				bufA.putDouble(va);
				bufB.putDouble(vb);
			}
			bufA.rewind();
			bufB.rewind();
			a.unmap(clCommandQueue, bufA);
			b.unmap(clCommandQueue, bufB);
			result = blas.reduce2(a, b, CLBlas.MergeOp.MUL, CLBlas.ReduceOp.ADD, result);
			assertEquals(dotP, blas.getReduceResultBlocking(result), Math.abs(dotP) * 10);
			
			if (size == performanceSize) {
				//measure execution speed
				clCommandQueue.finish();
				long timeA = System.nanoTime();
				for (int i=0; i<performanceCount; ++i) {
					result = blas.reduce2(a, b, CLBlas.MergeOp.MUL, CLBlas.ReduceOp.ADD, result);
				}
				long timeB = System.nanoTime();
				LOG.log(Level.INFO, "Time to execute one dot-product of vectors of size {0}: {1} ns", new Object[]{performanceSize, (timeB-timeA)/1000000.0});
				clCommandQueue.finish();
			}
			
			a.release();
			b.release();
		}
	}
	
	@Test
	public void testCompile() {
		//Initializes CLBlas for all supported data types
		CLBlas<?> b;
		b = CLBlas.get(settings, Float.class);
		b = CLBlas.get(settings, Double.class);
		b = CLBlas.get(settings, Integer.class);
		b = CLBlas.get(settings, Long.class);
	}
	
}

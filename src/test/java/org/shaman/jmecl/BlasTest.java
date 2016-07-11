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
import java.nio.IntBuffer;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.shaman.jmecl.utils.CLBlas;

import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Weiss
 */
public class BlasTest {
	private static final Logger LOG = Logger.getLogger(BlasTest.class.getName());
	
	private Platform clPlatform;
	private Device clDevice;
	private Context clContext;
	private CommandQueue clCommandQueue;
	private AssetManager assetManager;
	private OpenCLSettings settings;
	
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
	public void testCompile() {
		//Initializes CLBlas for all supported data types
		CLBlas<?> b;
		b = CLBlas.get(settings, Float.class);
		b = CLBlas.get(settings, Double.class);
		b = CLBlas.get(settings, Integer.class);
		b = CLBlas.get(settings, Long.class);
	}
	
	@Before
	public void setUp() {
		HeadlessContext hc = new HeadlessContext();
		assertTrue(hc.createOpenCLContext(false));
		clPlatform = hc.getClPlatform();
		clDevice = hc.getClDevice();
		clContext = hc.getClContext();
		clCommandQueue = clContext.createQueue(clDevice);
		assetManager = new DesktopAssetManager(true);
		settings = new OpenCLSettings(clContext, clCommandQueue, null, assetManager);
		LOG.info("OpenCL initialized");
	}
	
	@After
	public void tearDown() {
		OpenCLObjectManager.getInstance().deleteAllObjects();
		clCommandQueue.release();
		clContext.release();
		LOG.info("OpenCL released");
	}
	
}

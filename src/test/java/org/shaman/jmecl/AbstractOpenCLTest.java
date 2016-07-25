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
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 *
 * @author Sebastian Weiss
 */
public class AbstractOpenCLTest {
	private static final Logger LOG = Logger.getLogger(AbstractOpenCLTest.class.getName());
	
	protected Random rand;
	protected Platform clPlatform;
	protected Device clDevice;
	protected Context clContext;
	protected CommandQueue clCommandQueue;
	protected AssetManager assetManager;
	protected OpenCLSettings settings;
	
	protected void assertBufferEquals(Buffer buf, int offset, int... values) {
		ByteBuffer b = buf.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		for (int i=0; i<offset; ++i) {
			b.getInt();
		}
		for (int v : values) {
			assertEquals(v, b.getInt());
		}
		buf.unmap(clCommandQueue, b);
	}
	protected void assertBufferEquals(Buffer buf, int offset, long... values) {
		ByteBuffer b = buf.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		for (int i=0; i<offset; ++i) {
			b.getLong();
		}
		for (long v : values) {
			assertEquals(v, b.getLong());
		}
		buf.unmap(clCommandQueue, b);
	}
	protected void assertBufferEquals(Buffer buf, int offset, float... values) {
		ByteBuffer b = buf.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		for (int i=0; i<offset; ++i) {
			b.getFloat();
		}
		for (float v : values) {
			assertEquals(v, b.getFloat(), 0.00001);
		}
		buf.unmap(clCommandQueue, b);
	}
	protected void assertBufferEquals(Buffer buf, int offset, double... values) {
		ByteBuffer b = buf.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		for (int i=0; i<offset; ++i) {
			b.getDouble();
		}
		for (int i=0; i<values.length; ++i) {
			double v = values[i];
			assertEquals("Index " + i, v, b.getDouble(), 0.00001);
		}
		buf.unmap(clCommandQueue, b);
	}
	
	protected int randInt(int min, int max) {
		return rand.nextInt(max-min) + min;
	}
	protected double randDouble(double min, double max) {
		return rand.nextDouble()*(max-min) + min;
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
		rand = new Random();
	}
	
	@After
	public void tearDown() {
		OpenCLObjectManager.getInstance().deleteAllObjects();
		clCommandQueue.release();
		clContext.release();
		LOG.info("OpenCL released");
	}
}

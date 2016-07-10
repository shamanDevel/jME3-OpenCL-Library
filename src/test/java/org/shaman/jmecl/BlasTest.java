/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl;

import com.jme3.opencl.*;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

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
	
	public BlasTest() {
	}
	
	@Test
	public void testFill() {
		
	}
	
	@Before
	public void setUp() {
		HeadlessContext hc = new HeadlessContext();
		assertTrue(hc.createOpenCLContext(false));
		clPlatform = hc.getClPlatform();
		clDevice = hc.getClDevice();
		clContext = hc.getClContext();
		clCommandQueue = clContext.createQueue(clDevice);
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

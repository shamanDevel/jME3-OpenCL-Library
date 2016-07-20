/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl;

import com.jme3.opencl.Context;
import com.jme3.opencl.Device;
import com.jme3.opencl.Platform;
import com.jme3.opencl.lwjgl.LwjglDevice;
import com.jme3.opencl.lwjgl.LwjglPlatform;
import com.jme3.opencl.lwjgl.Utils;
import com.jme3.system.NativeLibraryLoader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.lwjgl.LWJGLException;
import org.lwjgl.opencl.*;

/**
 *
 * @author Sebastian Weiss
 */
public class HeadlessContext {
	private static final Logger LOG = Logger.getLogger(HeadlessContext.class.getName());
	
	private LwjglPlatform clPlatform;
	private LwjglDevice clDevice;
	private Context clContext;
	
	public boolean createOpenCLContext(boolean userSelection) {
		if (clContext != null) {
			LOG.severe("headless context already initialized");
			return false;
		}
		LOG.info("Initialize OpenCL with LWJGL2");
		
		NativeLibraryLoader.loadNativeLibrary("lwjgl", true);
        
        try {
            CL.create();
        } catch (LWJGLException ex) {
            LOG.log(Level.SEVERE, "Unable to initialize OpenCL", ex);
            return false;
        }
        
        //load platforms and devices
        StringBuilder platformInfos = new StringBuilder();
        ArrayList<LwjglPlatform> platforms = new ArrayList<>();
        for (CLPlatform p : CLPlatform.getPlatforms()) {
            platforms.add(new LwjglPlatform(p));
        }
        platformInfos.append("Available OpenCL platforms:");
        for (int i=0; i<platforms.size(); ++i) {
            LwjglPlatform platform = platforms.get(i);
            platformInfos.append("\n * Platform ").append(i+1);
            platformInfos.append("\n *   Name: ").append(platform.getName());
            platformInfos.append("\n *   Vendor: ").append(platform.getVendor());
            platformInfos.append("\n *   Version: ").append(platform.getVersion());
            platformInfos.append("\n *   Profile: ").append(platform.getProfile());
            platformInfos.append("\n *   Supports interop: ").append(platform.hasOpenGLInterop());
            List<LwjglDevice> devices = platform.getDevices();
            platformInfos.append("\n *   Available devices:");
            for (int j=0; j<devices.size(); ++j) {
                LwjglDevice device = devices.get(j);
                platformInfos.append("\n *    * Device ").append(j+1);
                platformInfos.append("\n *    *   Name: ").append(device.getName());
                platformInfos.append("\n *    *   Vendor: ").append(device.getVendor());
                platformInfos.append("\n *    *   Version: ").append(device.getVersion());
                platformInfos.append("\n *    *   Profile: ").append(device.getProfile());
                platformInfos.append("\n *    *   Compiler version: ").append(device.getCompilerVersion());
                platformInfos.append("\n *    *   Device type: ").append(device.getDeviceType());
                platformInfos.append("\n *    *   Compute units: ").append(device.getComputeUnits());
                platformInfos.append("\n *    *   Work group size: ").append(device.getMaxiumWorkItemsPerGroup());
                platformInfos.append("\n *    *   Global memory: ").append(device.getGlobalMemorySize()).append("B");
                platformInfos.append("\n *    *   Local memory: ").append(device.getLocalMemorySize()).append("B");
                platformInfos.append("\n *    *   Constant memory: ").append(device.getMaximumConstantBufferSize()).append("B");
                platformInfos.append("\n *    *   Supports double: ").append(device.hasDouble());
                platformInfos.append("\n *    *   Supports half floats: ").append(device.hasHalfFloat());
                platformInfos.append("\n *    *   Supports writable 3d images: ").append(device.hasWritableImage3D());
                platformInfos.append("\n *    *   Supports interop: ").append(device.hasOpenGLInterop());
            }
        }
        LOG.info(platformInfos.toString());
        
        //choose devices
		if (userSelection) {
			Scanner in = new Scanner(System.in);
			System.out.print("Select platform: "); System.out.flush();
			int platformID;
			do {
				platformID = in.nextInt()-1;
			} while (platformID<0 || platformID>=platforms.size());
			LwjglPlatform platform = platforms.get(platformID);
			int deviceID;
			System.out.print("Select device: "); System.out.flush();
			do {
				deviceID = in.nextInt()-1;
			} while (deviceID<0 || deviceID>=platform.getDevices().size());
			LwjglDevice device = platform.getDevices().get(deviceID);
			clDevice = device;
			clPlatform = platform;
		} else {
			//always choose the first one
			clPlatform = platforms.get(0);
			clDevice = clPlatform.getDevices().get(0);
			LOG.info(clPlatform+" and "+clDevice+" selected");
		}
		
        //create context
        try {
            CLContext c = CLContext.create(clPlatform.getPlatform(), Collections.singletonList(clDevice.getDevice()), Utils.errorBuffer);
            clContext = new com.jme3.opencl.lwjgl.LwjglContext(c, Collections.singletonList(clDevice));
        } catch (Exception ex) {
            LOG.log(Level.SEVERE, "Unable to create OpenCL context", ex);
            return false;
        }
		Utils.checkError(Utils.errorBuffer, "clCreateContext");
        
        LOG.info("OpenCL context created");
		return true;
	}

	public Platform getClPlatform() {
		return clPlatform;
	}

	public Device getClDevice() {
		return clDevice;
	}

	public Context getClContext() {
		return clContext;
	}
	
	
}

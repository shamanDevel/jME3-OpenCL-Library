/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.test;

import com.jme3.opencl.Device;
import com.jme3.opencl.Platform;
import com.jme3.opencl.PlatformChooser;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author Sebastian Weiss
 */
public class UserPlatformChooser implements PlatformChooser{

	@Override
	public List<? extends Device> chooseDevices(List<? extends Platform> platforms) {
		Scanner in = new Scanner(System.in);
		
		StringBuilder platformInfos = new StringBuilder();
		platformInfos.append("Available OpenCL platforms:");
        for (int i=0; i<platforms.size(); ++i) {
            Platform platform = platforms.get(i);
            platformInfos.append("\n * Platform ").append(i+1);
            platformInfos.append("\n *   Name: ").append(platform.getName());
            platformInfos.append("\n *   Vendor: ").append(platform.getVendor());
            platformInfos.append("\n *   Version: ").append(platform.getVersion());
            platformInfos.append("\n *   Profile: ").append(platform.getProfile());
            platformInfos.append("\n *   Supports interop: ").append(platform.hasOpenGLInterop());
            List<? extends Device> devices = platform.getDevices();
            platformInfos.append("\n *   Available devices:");
            for (int j=0; j<devices.size(); ++j) {
                Device device = devices.get(j);
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
        System.out.println(platformInfos);
        
        //choose devices
		Platform platform;
		if (platforms.size() == 1) {
			platform = platforms.get(0);
			System.out.println("Selected platform 1");
		} else {
			System.out.println("Select platform: "); System.out.flush();
			int platformID;
			do {
				platformID = in.nextInt()-1;
			} while (platformID<0 || platformID>=platforms.size());
			platform = platforms.get(platformID);
		}
		Device device;
		if (platform.getDevices().size() == 1) {
			device = platform.getDevices().get(0);
			System.out.println("Selected device 1");
		} else {
			int deviceID;
			System.out.println("Select device: "); System.out.flush();
			do {
				deviceID = in.nextInt()-1;
			} while (deviceID<0 || deviceID>=platform.getDevices().size());
			device = platform.getDevices().get(deviceID);
		}
		return Collections.singletonList(device);
	}
	
}

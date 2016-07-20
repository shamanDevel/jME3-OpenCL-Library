/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl;

import com.jme3.asset.AssetManager;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.opencl.ProgramCache;
import java.util.Objects;

/**
 *
 * @author Sebastian Weiss
 */
public class OpenCLSettings {
	
	private final Context clContext;
	private final CommandQueue clCommandQueue;
	private final ProgramCache programCache;
	private final AssetManager assetManager;

	public OpenCLSettings(Context clContext, CommandQueue clCommandQueue, 
			ProgramCache programCache, AssetManager assetManager) {
		this.clContext = clContext;
		this.clCommandQueue = clCommandQueue;
		this.programCache = programCache!=null ? programCache : new ProgramCache();
		this.assetManager = assetManager;
	}

	public Context getClContext() {
		return clContext;
	}

	public CommandQueue getClCommandQueue() {
		return clCommandQueue;
	}

	public ProgramCache getProgramCache() {
		return programCache;
	}

	public AssetManager getAssetManager() {
		return assetManager;
	}

	@Override
	public int hashCode() {
		int hash = 3;
		hash = 43 * hash + Objects.hashCode(this.clContext);
		hash = 43 * hash + Objects.hashCode(this.clCommandQueue);
		return hash;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		final OpenCLSettings other = (OpenCLSettings) obj;
		if (!Objects.equals(this.clContext, other.clContext)) {
			return false;
		}
		if (!Objects.equals(this.clCommandQueue, other.clCommandQueue)) {
			return false;
		}
		return true;
	}

}

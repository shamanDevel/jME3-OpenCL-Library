/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.opencl.*;

/**
 * Collection of utitlity and blas-level-1 routines on primitive types.
 * @author Sebastian Weiss
 */
public class CLBlas {
	
	private final Context clContext;
	private final CommandQueue clCommandQueue;
	
	
	private CLBlas() {
		clContext = null;
		clCommandQueue = null;
	}
	
}

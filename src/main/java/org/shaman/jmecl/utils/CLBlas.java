/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.opencl.*;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;

/**
 * Collection of utitlity and blas-level-1 routines on primitive types.
 * @author Sebastian Weiss
 * @param <T> the number type, Float, Double, Integer and Long are supported
 */
public final class CLBlas<T extends Number> {
	private static final Logger LOG = Logger.getLogger(CLBlas.class.getName());
	private static final String FILE = "org/shaman/jmecl/utils/CLBlas.cl";
	
	private static final Map<Class<? extends Number>, CLBlas<? extends Number>> instances
			= new HashMap<>();
	private static class ElementSpecs {
		private final int elementSize;
		private final String clType;
		private ElementSpecs(int elementSize, String clType) {
			this.elementSize = elementSize;
			this.clType = clType;
		}
	}
	private static final Map<Class<? extends Number>, ElementSpecs> specs
			= new HashMap<>();
	static {
		specs.put(Float.class, new ElementSpecs(4, "float"));
		specs.put(Double.class, new ElementSpecs(8, "double"));
		specs.put(Integer.class, new ElementSpecs(4, "int"));
		specs.put(Long.class, new ElementSpecs(8, "long"));
	}
	
	private final Context clContext;
	private final CommandQueue clCommandQueue;
	
	private final Class<T> elementClass;
	private final int elementSize;
	
	private final Program program;
	private final Kernel fillKernel;
	private final Kernel axpyKernel;
	
	private CLBlas(OpenCLSettings settings, Class<T> numberType) {
		clContext = settings.getClContext();
		clCommandQueue = settings.getClCommandQueue();
		
		elementClass = numberType;
		ElementSpecs es = specs.get(numberType);
		if (es == null) {
			throw new UnsupportedOperationException("Unsupported number type "+numberType);
		}
		elementSize = es.elementSize;
		
		String cacheID = CLBlas.class.getName() + "-" + numberType.getSimpleName();
		Program p = settings.getProgramCache().loadFromCache(cacheID);
		if (p == null) {
			StringBuilder includes = new StringBuilder();
			includes.append("#define TYPE ").append(es.clType).append("\n\n");
			p = clContext.createProgramFromSourceFilesWithInclude(
					settings.getAssetManager(), includes.toString(), FILE);
			//settings.getProgramCache().saveToCache(cacheID, p);
		}
		program = p;
		p.build();
		fillKernel = p.createKernel("Fill");
		axpyKernel = p.createKernel("AXPY");
	}
	
	public static <T extends Number> CLBlas<T> get(OpenCLSettings settings, Class<T> numberType) {
		@SuppressWarnings("unchecked")
		CLBlas<T> blas = (CLBlas<T>) instances.get(numberType);
		if (blas == null) {
			blas = new CLBlas<>(settings, numberType);
			instances.put(numberType, blas);
		}
		return blas;
	}
	
	public Event fill(Buffer b, T val, long size, long offset, long step) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		return fillKernel.Run1(clCommandQueue, ws, b, val, offset, step);
	}
	public Event fill(Buffer b, T val, long size) {
		return fill(b, val, size, 0, 1);
	}
	public Event fill(Buffer b, T val) {
		return fill(b, val, b.getSize()/elementSize);
	}
	
	/**
	 * Computes {@code dest[i] = a*x[i] + y[i]}.
	 * @param x
	 * @param a
	 * @param y
	 * @param dest
	 * @param size
	 * @param offsetX
	 * @param offsetY
	 * @param offsetDest
	 * @param stepX
	 * @param stepY
	 * @param stepDest
	 * @return 
	 */
	public Event axpy(T a, Buffer x, Buffer y, Buffer dest, 
			long size, long offsetX, long offsetY, long offsetDest,
			long stepX, long stepY, long stepDest) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		return axpyKernel.Run1(clCommandQueue, ws, a, x, y, dest, 
				offsetX, offsetY, offsetDest, stepX, stepY, stepDest);
	}
	public Event axpy(T a, Buffer x, Buffer y, Buffer dest, long size) {
		return axpy(a, x, y, dest, size, 0, 0, 0, 1, 1, 1);
	}
	public Event axpy(T a, Buffer x, Buffer y, Buffer dest) {
		long size = Math.min(Math.min(dest.getSize(), x.getSize()), y.getSize());
		size /= elementSize;
		return axpy(a, x, y, dest, size);
	}
	
	public static enum MapOp {
		SET,
		ADD,
		SUB,
		MUL,
		SUB_INV,
		DIV,
		DIV_INV,
		ABS,
		EXP,
		LOG,
		POW,
		POW_INV
	}
	
	public Event map(Buffer b, MapOp op, T arg, Buffer dest, 
			long size, long offsetB, long offsetDest, long stepB, long stepDest) {
		throw new UnsupportedOperationException("not supported yet");
	}
	public Event map(Buffer b, MapOp op, T arg, Buffer dest, long size) {
		return map(b, op, arg, dest, size,  0, 0, 1, 1);
	}
	public Event map(Buffer b, MapOp op, T arg, Buffer dest) {
		long size = Math.min(b.getSize(), dest.getSize()) / elementSize;
		return map(b, op, arg, dest, size);
	}
	
	public static enum PreReduceOp {
		NONE,
		ABS,
		SQUARE
	}
	public static enum ReduceOp {
		ADD,
		MUL,
		MIN,
		MAX
	}
	public static enum MergeOp {
		ADD,
		SUB,
		MUL,
		MIN,
		MAX
	}
	
	public static class ReduceResult {
		private Event event;
		private Buffer result;

		public ReduceResult() {
		}

		public Event getEvent() {
			return event;
		}

		public Buffer getResult() {
			return result;
		}
		
	}
	
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			long size, long offset, long step, ReduceResult result) {
		throw new UnsupportedOperationException("not supported yet");
	}
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			long size, ReduceResult result) {
		return reduce(b, preReduceOp, reduceOp, size, 0, 1, result);
	}
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			ReduceResult result) {
		return reduce(b, preReduceOp, reduceOp, b.getSize()/elementSize, result);
	}
	
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			PreReduceOp preReduceOp, ReduceOp reduceOp,
			long size, long offsetA, long offsetB, long stepA, long stepB,
			ReduceResult result) {
		throw new UnsupportedOperationException("not supported yet");
	}
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			PreReduceOp preReduceOp, ReduceOp reduceOp, long size, ReduceResult result) {
		return reduce2(a, b, mergeOp, preReduceOp, reduceOp, size, 0, 0, 1, 1, result);
	}
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			PreReduceOp preReduceOp, ReduceOp reduceOp,	ReduceResult result) {
		long size = Math.min(a.getSize(), b.getSize()) / elementSize;
		return reduce2(a, b, mergeOp, preReduceOp, reduceOp, size, result);
	}
	
	public T getReduceResultBlocking(ReduceResult result) {
		throw new UnsupportedOperationException("not supported yet");
	}
}

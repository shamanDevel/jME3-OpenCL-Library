/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.opencl.*;
import java.nio.ByteBuffer;
import java.util.AbstractMap;
import java.util.EnumMap;
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
	
	private static final Map<OpenCLSettings, Map<Class<? extends Number>, CLBlas<? extends Number>>> instances
			= new HashMap<>();
	private static interface ElementGetter {
		Number get(ByteBuffer b);
	}
	private static class ElementSpecs {
		private final int elementSize;
		private final String clType;
		private final boolean floatType;
		private final String clTypeMax;
		private final String clTypeMin;
		private final ElementGetter getter;
		private ElementSpecs(int elementSize, String clType, boolean floatType, 
				String clTypeMax, String clTypeMin, ElementGetter getter) {
			this.elementSize = elementSize;
			this.clType = clType;
			this.floatType = floatType;
			this.clTypeMax = clTypeMax;
			this.clTypeMin = clTypeMin;
			this.getter = getter;
		}
	}
	private static final Map<Class<? extends Number>, ElementSpecs> specs
			= new HashMap<>();
	static {
		specs.put(Float.class, new ElementSpecs(4, "float", true, "FLT_MAX", "-FLT_MAX", new ElementGetter() {
			@Override
			public Number get(ByteBuffer b) {
				return b.getFloat();
			}
		}));
		specs.put(Double.class, new ElementSpecs(8, "double", true, "DBL_MAX", "-DBL_MAX", new ElementGetter() {
			@Override
			public Number get(ByteBuffer b) {
				return b.getDouble();
			}
		}));
		specs.put(Integer.class, new ElementSpecs(4, "int", false, "INT_MAX", "INT_MIN", new ElementGetter() {
			@Override
			public Number get(ByteBuffer b) {
				return b.getInt();
			}
		}));
		specs.put(Long.class, new ElementSpecs(8, "long", false, "LONG_MAX", "LONG_MIN", new ElementGetter() {
			@Override
			public Number get(ByteBuffer b) {
				return b.getLong();
			}
		}));
	}
	
	private final Context clContext;
	private final CommandQueue clCommandQueue;
	
	private final Class<T> elementClass;
	private final int elementSize;
	private final ElementGetter getter;
	
	private final int workgroupSize;
	private final int workgroups;
	private final Program program;
	private final Kernel fillKernel;
	private final Kernel axpyKernel;
	private final EnumMap<MapOp, Kernel> mapKernels;
	private final EnumMap<PreReduceOp, EnumMap<ReduceOp, Kernel>> reduceKernels;
	private final EnumMap<MergeOp, EnumMap<ReduceOp, Kernel>> reduce2Kernels;
	private Buffer tmpMem;
	
	private CLBlas(OpenCLSettings settings, Class<T> numberType) {
		clContext = settings.getClContext();
		clCommandQueue = settings.getClCommandQueue();
		workgroupSize = (int) Math.min(256, clCommandQueue.getDevice().getMaxiumWorkItemsPerGroup());
		workgroups = clCommandQueue.getDevice().getComputeUnits(); // *32
		
		elementClass = numberType;
		ElementSpecs es = specs.get(numberType);
		if (es == null) {
			throw new UnsupportedOperationException("Unsupported number type "+numberType);
		}
		elementSize = es.elementSize;
		getter = es.getter;
		
		String cacheID = CLBlas.class.getName() + "-" + numberType.getSimpleName();
		Program p = settings.getProgramCache().loadFromCache(cacheID);
		if (p == null) {
			StringBuilder includes = new StringBuilder();
			if (elementClass == Double.class) {
				includes.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
			}
			includes.append("#define TYPE ").append(es.clType).append("\n");
			includes.append("#define TYPE_MIN ").append(es.clTypeMin).append("\n");
			includes.append("#define TYPE_MAX ").append(es.clTypeMax).append("\n");
			includes.append("#define IS_FLOAT_TYPE ").append(es.floatType ? 1 : 0).append("\n\n");
			p = clContext.createProgramFromSourceFilesWithInclude(
					settings.getAssetManager(), includes.toString(), FILE);
			//settings.getProgramCache().saveToCache(cacheID, p);
		}
		program = p;
		p.register();
		p.build();
		fillKernel = p.createKernel("Fill").register();
		axpyKernel = p.createKernel("AXPY").register();
		mapKernels = new EnumMap<>(MapOp.class);
		for (MapOp op : MapOp.values()) {
			mapKernels.put(op, p.createKernel("Map_" + op.name()).register());
		}
		reduceKernels = new EnumMap<>(PreReduceOp.class);
		for (PreReduceOp op1 : PreReduceOp.values()) {
			EnumMap<ReduceOp, Kernel> map = new EnumMap<>(ReduceOp.class);
			for (ReduceOp op2 : ReduceOp.values()) {
				map.put(op2, p.createKernel("Reduce_"+op1.name()+"_"+op2.name()).register());
			}
			reduceKernels.put(op1, map);
		}
		reduce2Kernels = new EnumMap<>(MergeOp.class);
		for (MergeOp op1 : MergeOp.values()) {
			EnumMap<ReduceOp, Kernel> map = new EnumMap<>(ReduceOp.class);
			for (ReduceOp op2 : ReduceOp.values()) {
				map.put(op2, p.createKernel("Reduce2_"+op1.name()+"_"+op2.name()).register());
			}
			reduce2Kernels.put(op1, map);
		}
	}
	
	/**
	 * Returns the blas instance for the specified number type and opencl settings.
	 * Supported number types: Integer, Long, Float, Double.
	 * @param <T> the number type
	 * @param settings the opencl settings: context, command queue, asset manager and program cache
	 * @param numberType the class of the number type
	 * @return the blas instance
	 */
	public static <T extends Number> CLBlas<T> get(OpenCLSettings settings, Class<T> numberType) {
		Map<Class<? extends Number>, CLBlas<? extends Number>> map = instances.get(settings);
		if (map == null) {
			map = new HashMap<>();
			instances.put(settings, map);
		}
		@SuppressWarnings("unchecked")
		CLBlas<T> blas = (CLBlas<T>) map.get(numberType);
		if (blas == null) {
			blas = new CLBlas<>(settings, numberType);
			map.put(numberType, blas);
		}
		return blas;
	}
	
	private int nextPow2 (int x) 
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}
	
	private Buffer ensureBufferSize(Buffer buffer, long size) {
		if (buffer == null) {
			return clContext.createBuffer(size).register();
		} else {
			if (buffer.getSize() < size) {
				buffer.release();
				return clContext.createBuffer(size).register();
			} else {
				return buffer;
			}
		}
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
		EXP,  //will run with double-precision for integer types
		LOG,  // "
		POW,  // "
		POW_INV  // "
	}
	
	public Event map(Buffer b, MapOp op, T arg, Buffer dest, 
			long size, long offsetB, long offsetDest, long stepB, long stepDest) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		Kernel kernel = mapKernels.get(op);
		return kernel.Run1(clCommandQueue, ws, b, arg, dest, offsetB, offsetDest, stepB, stepDest);
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
	
	private void getReduceWorkSize (int bufferSize, int[] result) //result: std::size_t* numWorkGroups, std::size_t* workGroupSize
	{
		int workGroupSize = (bufferSize < workgroupSize) ? nextPow2 (bufferSize) : workgroupSize;
		
		int numWorkGroups = (bufferSize + ((workGroupSize) - 1)) / (workGroupSize);
		numWorkGroups = Math.min(workgroups, numWorkGroups);
		
		result[0] = workGroupSize;
		result[1] = numWorkGroups;
	}

	
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			long size, long offset, long step, ReduceResult result) {
		if (size > Integer.MAX_VALUE) {
			throw new IllegalArgumentException("buffer is too big, only 2^31 elements supported");
		}
		
		if (result == null) {
			result = new ReduceResult();
		}
		result.result = ensureBufferSize(result.result, elementSize);
		
		int[] sizes = new int[2];
		getReduceWorkSize((int) size, sizes);
		int workGroupSize = sizes[0];
		int numWorkGroups = sizes[1];
		int globalWorkSize = numWorkGroups * workGroupSize;

		tmpMem = ensureBufferSize(tmpMem, elementSize * numWorkGroups);
		
		Kernel kernelOp1 = reduceKernels.get(preReduceOp).get(reduceOp);
		Kernel kernelOp2 = reduceKernels.get(PreReduceOp.NONE).get(reduceOp);
		
		kernelOp1.Run2NoEvent(clCommandQueue, new Kernel.WorkSize(globalWorkSize), new Kernel.WorkSize(workGroupSize), 
					b, new Kernel.LocalMem(elementSize), (int) size, tmpMem, (int) offset, (int) step);
		
		size = numWorkGroups;
		while (size > 1) {
			getReduceWorkSize((int) size, sizes);
			workGroupSize = sizes[0];
			numWorkGroups = sizes[1];
			globalWorkSize = numWorkGroups * workGroupSize;
			
			kernelOp2.Run2NoEvent(clCommandQueue, new Kernel.WorkSize(globalWorkSize), new Kernel.WorkSize(workGroupSize), 
						tmpMem, new Kernel.LocalMem(elementSize), (int) size, tmpMem, (int) 0, (int) 1);
			
			size = numWorkGroups;
		}
		
		result.event = tmpMem.copyToAsync(clCommandQueue, result.result, elementSize).register();
		return result;
	}
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			long size, ReduceResult result) {
		return reduce(b, preReduceOp, reduceOp, size, 0, 1, result);
	}
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			ReduceResult result) {
		return reduce(b, preReduceOp, reduceOp, b.getSize()/elementSize, result);
	}
	
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp, ReduceOp reduceOp,
			long size, long offsetA, long offsetB, long stepA, long stepB,
			ReduceResult result) {
		if (size > Integer.MAX_VALUE) {
			throw new IllegalArgumentException("buffer is too big, only 2^31 elements supported");
		}
		
		if (result == null) {
			result = new ReduceResult();
		}
		result.result = ensureBufferSize(result.result, elementSize);
		
		int[] sizes = new int[2];
		getReduceWorkSize((int) size, sizes);
		int workGroupSize = sizes[0];
		int numWorkGroups = sizes[1];
		int globalWorkSize = numWorkGroups * workGroupSize;

		tmpMem = ensureBufferSize(tmpMem, elementSize * numWorkGroups);
		
		Kernel kernelOp1 = reduce2Kernels.get(mergeOp).get(reduceOp);
		Kernel kernelOp2 = reduceKernels.get(PreReduceOp.NONE).get(reduceOp);
		
		kernelOp1.Run2NoEvent(clCommandQueue, new Kernel.WorkSize(globalWorkSize), new Kernel.WorkSize(workGroupSize), 
					a, b, new Kernel.LocalMem(elementSize), (int) size, tmpMem, (int) offsetA, (int) stepA, (int) offsetB, (int) stepB);
		
		size = numWorkGroups;
		while (size > 1) {
			getReduceWorkSize((int) size, sizes);
			workGroupSize = sizes[0];
			numWorkGroups = sizes[1];
			globalWorkSize = numWorkGroups * workGroupSize;
			
			kernelOp2.Run2NoEvent(clCommandQueue, new Kernel.WorkSize(globalWorkSize), new Kernel.WorkSize(workGroupSize), 
						tmpMem, new Kernel.LocalMem(elementSize), (int) size, tmpMem, (int) 0, (int) 1);
			
			size = numWorkGroups;
		}
		
		result.event = tmpMem.copyToAsync(clCommandQueue, result.result, elementSize).register();
		return result;
	}
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			ReduceOp reduceOp, long size, ReduceResult result) {
		return reduce2(a, b, mergeOp, reduceOp, size, 0, 0, 1, 1, result);
	}
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			ReduceOp reduceOp,	ReduceResult result) {
		long size = Math.min(a.getSize(), b.getSize()) / elementSize;
		return reduce2(a, b, mergeOp, reduceOp, size, result);
	}
	
	public ReduceResult dotProduct(Buffer a, Buffer b, ReduceResult result) {
		return reduce2(a, b, MergeOp.MUL, ReduceOp.ADD, result);
	}
	
	@SuppressWarnings("unchecked")
	public T getReduceResultBlocking(ReduceResult result) {
		ByteBuffer buf = result.result.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		Number n = getter.get(buf);
		result.result.unmap(clCommandQueue, buf);
		return (T) n;
	}
}

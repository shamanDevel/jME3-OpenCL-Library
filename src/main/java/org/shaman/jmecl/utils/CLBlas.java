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
 * <p>
 * All operations take the following arguments:
 * <ul>
 *  <li>{@code size} the size or number of elements that the operation should process </li>
 *  <li>{@code step} the step size in the array, must be non-zero </li>
 *  <li>{@code offset} the offset into the array, must be non-negative </li>
 * </ul>
 * The passed buffers are adressed as they would be arrays of type {@code T}:
 * {@code T val = buffer[index * step + offset]} with {@code index} ranging from {@code 0}
 * to {@code size-1}. <br>
 * It must be ensured that the passed buffer large enough so that no element outside
 * the bounds is accessed.
 * 
 * 
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
		specs.put(Byte.class, new ElementSpecs(4, "char", false, "CHAR_MAX", "CHAR_MIN", new ElementGetter() {
			@Override
			public Number get(ByteBuffer b) {
				return b.get();
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
	private final Map<Integer, Kernel> reorderKernels;
	private final Kernel fillIndicesKernel;
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
		int reorderComponents[] = {1, 2, 3, 4};
		if (p == null) {
			StringBuilder includes = new StringBuilder();
			if (elementClass == Double.class) {
				includes.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
			}
			includes.append("#define TYPE ").append(es.clType).append("\n");
			for (int c : reorderComponents) {
				if (c>1) {
					includes.append("#define TYPE").append(c).append(" ").append(es.clType).append(c).append("\n");
				}
			}
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
		reorderKernels = new HashMap<>();
		for (int c : reorderComponents) {
			reorderKernels.put(c, p.createKernel("Reorder_"+c).register());
		}
		fillIndicesKernel = p.createKernel("FillIndices").register();
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
	
	/**
	 * Returns the number of bytes needed for the specified number class.
	 * For the supported number tpyes, this method return:
	 * <ul>
	 *  <li>Float: 4 </li>
	 *  <li>Double: 8 </li>
	 *  <li>Integer: 4 </li>
	 *  <li>Long: 8 </li>
	 * </ul>
	 * @return the count of bytes needed to represent the number class
	 * @see #get(org.shaman.jmecl.OpenCLSettings, java.lang.Class) 
	 */
	public int getElementSize() {
		return elementSize;
	}
	
	/**
	 * Fills the specified buffer with a constant value.
	 * @param b the buffer to fill
	 * @param val the value
	 * @param size the size/count of elements to fill
	 * @param offset the offset into the buffer
	 * @param step the step size
	 * @return the event object, must be released manually
	 */
	public Event fill(Buffer b, T val, long size, long offset, long step) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		return fillKernel.Run1(clCommandQueue, ws, b, val, offset, step);
	}
	/**
	 * Fills a part of the buffer.
	 * Convenient method, calls {@code fill(b, val, size, 0, 1)}
	 * @param b the buffer to fill
	 * @param val the value
	 * @param size the size
	 * @return the event, must be released manually
	 * @see #fill(com.jme3.opencl.Buffer, java.lang.Number, long, long, long) 
	 */
	public Event fill(Buffer b, T val, long size) {
		return fill(b, val, size, 0, 1);
	}
	/**
	 * Fills the whole buffer.
	 * Convenient method, calls {@code fill(b, val, b.getSize()/getElementSize(), 0, 1}
	 * @param b the buffer to fill
	 * @param val the value
	 * @return the event, must be released manually
	 * @see #fill(com.jme3.opencl.Buffer, java.lang.Number, long, long, long) 
	 */
	public Event fill(Buffer b, T val) {
		return fill(b, val, b.getSize()/elementSize);
	}
	
	/**
	 * Fills a part of the buffer with an increasing sequence.
	 * It assigns each element a value with the following equation:
	 * {@code x[idx] = start + idx * step}.
	 * @param x the destination buffer
	 * @param start the start value
	 * @param step the step value
	 * @param size the number of elements to process
	 * @return the event object
	 */
	public Event fillIndices(Buffer x, T start, T step, long size) {
		return fillIndicesKernel.Run1(clCommandQueue, new Kernel.WorkSize(size), x, start, step);
	}
	
	/**
	 * Fills the whole buffer with an increasing sequence.
	 * It assigns each element a value with the following equation:
	 * {@code x[idx] = start + idx * step}.
	 * @param x the destination buffer
	 * @param start the start value
	 * @param step the step value
	 * @return the event object
	 * @see #fillIndices(com.jme3.opencl.Buffer, java.lang.Number, java.lang.Number, long) 
	 */
	public Event fillIndices(Buffer x, T start, T step) {
		return fillIndices(x, start, step, x.getSize()/elementSize);
	}
	
	/**
	 * Reorders the buffers with the specified index buffers.
	 * This is used to reorder other buffers after only an index buffer has
	 * been sorted.
	 * @param indices the index buffer, type uint
	 * @param src the source buffer
	 * @param dest the destination buffer
	 * @param components the number of components per entry (like float, float2, float4)
	 * @param size the count of elements
	 * @return the event, must be released manually
	 */
	public Event reorder(Buffer indices, Buffer src, Buffer dest, int components, long size) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		Kernel k = reorderKernels.get(components);
		return k.Run1(clCommandQueue, ws, indices, src, dest);
	}
	
	/**
	 * Computes {@code dest[i] = a*x[i] + y[i]}.
	 * @param a the scalar value multiplied with x
	 * @param x the first buffer
	 * @param y the second buffer
	 * @param dest the destination buffer
	 * @param size the number of elements to process
	 * @param offsetX offset into the x buffer
	 * @param offsetY offset into the y buffer
	 * @param offsetDest offset into the dest buffer
	 * @param stepX step size in the x buffer
	 * @param stepY step size in the y buffer
	 * @param stepDest step size in the dest buffer
	 * @return the event, must be released manually
	 */
	public Event axpy(T a, Buffer x, Buffer y, Buffer dest, 
			long size, long offsetX, long offsetY, long offsetDest,
			long stepX, long stepY, long stepDest) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		return axpyKernel.Run1(clCommandQueue, ws, a, x, y, dest, 
				offsetX, offsetY, offsetDest, stepX, stepY, stepDest);
	}
	/**
	 * Computes {@code dest[i] = a*x[i] + y[i]} on a part of the buffers.
	 * Calls {@code axpy(a, x, y, dest, size, 0, 0, 0, 1, 1, 1)}.
	 * @param a the scalar value multiplied with x
	 * @param x the first buffer
	 * @param y the second buffer
	 * @param dest the destination buffer
	 * @param size the number of elements to process
	 * @return the event, must be released manually
	 * @see #axpy(java.lang.Number, com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, long, long, long, long, long, long, long) 
	 */
	public Event axpy(T a, Buffer x, Buffer y, Buffer dest, long size) {
		return axpy(a, x, y, dest, size, 0, 0, 0, 1, 1, 1);
	}
	/**
	 * Computes {@code dest[i] = a*x[i] + y[i]} on the whole buffer.
	 * @param a the scalar value multiplied with x
	 * @param x the first buffer
	 * @param y the second buffer
	 * @param dest the destination buffer
	 * @return the event, must be released manually
	 * @see #axpy(java.lang.Number, com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, long, long, long, long, long, long, long) 
	 */
	public Event axpy(T a, Buffer x, Buffer y, Buffer dest) {
		long size = Math.min(Math.min(dest.getSize(), x.getSize()), y.getSize());
		size /= elementSize;
		return axpy(a, x, y, dest, size);
	}
	
	/**
	 * The available mapping operations for 
	 * {@link #map(com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MapOp, java.lang.Number, com.jme3.opencl.Buffer, long, long, long, long, long) }.
	 * Let {@code x} be the input from the input buffer, {@code a} the additional
	 * argument and {@code z} be the output.
	 */
	public static enum MapOp {
		/**
		 * {@code z=a}
		 */
		SET,
		/**
		 * {@code z=x+a}
		 */
		ADD,
		/**
		 * {@code z=x-a}
		 */
		SUB,
		/**
		 * {@code z=x*a}
		 */
		MUL,
		/**
		 * {@code z=a-x}
		 */
		SUB_INV,
		/**
		 * {@code z=a/x}
		 */
		DIV,
		/**
		 * {@code z=x/a}
		 */
		DIV_INV,
		/**
		 * {@code z=abs(x)}
		 */
		ABS,
		/**
		 * {@code z=exp(x)}, computed with float-precision on integer types
		 */
		EXP,
		/**
		 * {@code z=log(x)/log(a)}, computed with float-precision on integer types
		 */
		LOG,
		/**
		 * {@code z=x^a}, computed with float-precision on integer types
		 */
		POW,
		/**
		 * {@code z=a^x}, computed with float-precision on integer types
		 */
		POW_INV
	}
	
	/**
	 * Performs a map operation / transformation on the specified buffer.
	 * @param b the buffer
	 * @param op the map operation
	 * @param arg an additional argument to the operation
	 * @param dest the destination buffer
	 * @param size the count of elements to process
	 * @param offsetB offset into the source buffer
	 * @param offsetDest offset into the destination buffer
	 * @param stepB step size in the source buffer
	 * @param stepDest step size in the destination buffer
	 * @return the event, must be released manually
	 * @see MapOp
	 */
	public Event map(Buffer b, MapOp op, T arg, Buffer dest, 
			long size, long offsetB, long offsetDest, long stepB, long stepDest) {
		Kernel.WorkSize ws = new Kernel.WorkSize(size);
		Kernel kernel = mapKernels.get(op);
		return kernel.Run1(clCommandQueue, ws, b, arg, dest, offsetB, offsetDest, stepB, stepDest);
	}
	/**
	 * Performs a map operation / transformation on a part of the buffer
	 * @param b the input buffer
	 * @param op the map operation
	 * @param arg an additional argument
	 * @param dest the destination buffer
	 * @param size the count of elements to process
	 * @return the event, must be released manually
	 * @see #map(com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MapOp, java.lang.Number, com.jme3.opencl.Buffer, long, long, long, long, long) 
	 * @see MapOp
	 */
	public Event map(Buffer b, MapOp op, T arg, Buffer dest, long size) {
		return map(b, op, arg, dest, size,  0, 0, 1, 1);
	}
	/**
	 * Performs a map operation / transformation on the whole buffer
	 * @param b the input buffer
	 * @param op the map operation
	 * @param arg an additional argument
	 * @param dest the destination buffer
	 * @return the event, must be released manually
	 * @see #map(com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MapOp, java.lang.Number, com.jme3.opencl.Buffer, long, long, long, long, long) 
	 * @see MapOp
	 */
	public Event map(Buffer b, MapOp op, T arg, Buffer dest) {
		long size = Math.min(b.getSize(), dest.getSize()) / elementSize;
		return map(b, op, arg, dest, size);
	}
	
	/**
	 * Operation applied to each element before it is reduced.
	 * @see #reduce(com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.PreReduceOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult)
	 */
	public static enum PreReduceOp {
		/**
		 * No modification, pass through
		 */
		NONE,
		/**
		 * The absolute value is taken
		 */
		ABS,
		/**
		 * The input is squared
		 */
		SQUARE
	}
	/**
	 * The reduce operation
	 * @see #reduce(com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.PreReduceOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 * @see #reduce2(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MergeOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 */
	public static enum ReduceOp {
		ADD,
		MUL,
		MIN,
		MAX
	}
	/**
	 * Operation to merge two buffers together before the result is reduced.
	 * @see #reduce2(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MergeOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 */
	public static enum MergeOp {
		ADD,
		SUB,
		MUL,
		MIN,
		MAX
	}
	
	/**
	 * Structure storing the result of a reduce operation.
	 * This class captures the asynchron result of the operation in a buffer.
	 * <br>
	 * The buffer only stores one single value. You can either pass it directly
	 * as an argument to another kernel, or read to value by {@link #getReduceResultBlocking(org.shaman.jmecl.utils.CLBlas.ReduceResult) }.
	 * <br>
	 * Instances of this class can be used multiple times, just pass it again
	 * to a reduce method.
	 */
	public static class ReduceResult {
		private Event event;
		private Buffer result;

		public ReduceResult() {
		}

		/**
		 * Returns the event indicating when the reduce operation is done
		 * @return the event, automatically released
		 */
		public Event getEvent() {
			return event;
		}

		/**
		 * The buffer storing the single return value.
		 * @return the buffer with the return value, automatically released
		 */
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

	/**
	 * Performs a reduce operation.
	 * @param b the input buffer
	 * @param preReduceOp an operation applied before the reduce operation is performed
	 * @param reduceOp the reduce operation
	 * @param size the count of elements to process
	 * @param offset the offset into the input buffer
	 * @param step the step size in the input buffer
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 */
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
					b, new Kernel.LocalMemPerElement(elementSize), (int) size, tmpMem, (int) offset, (int) step);
                clCommandQueue.finish();
		
		size = numWorkGroups;
		while (size > 1) {
			getReduceWorkSize((int) size, sizes);
			workGroupSize = sizes[0];
			numWorkGroups = sizes[1];
			globalWorkSize = numWorkGroups * workGroupSize;
			
			kernelOp2.Run2NoEvent(clCommandQueue, new Kernel.WorkSize(globalWorkSize), new Kernel.WorkSize(workGroupSize), 
						tmpMem, new Kernel.LocalMemPerElement(elementSize), (int) size, tmpMem, (int) 0, (int) 1);
                        clCommandQueue.finish();
			
			size = numWorkGroups;
		}
		
		result.event = tmpMem.copyToAsync(clCommandQueue, result.result, elementSize).register();
                clCommandQueue.finish();
		return result;
	}
	/**
	 * Performs a reduce operation on a part of the buffer.
	 * Calls {@code reduce(b, preReduceOp, reduceOp, size, 0, 1, result)}.
	 * @param b the input buffer
	 * @param preReduceOp an operation applied before the reduce operation is performed
	 * @param reduceOp the reduce operation
	 * @param size the count of elements to process
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 */
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			long size, ReduceResult result) {
		return reduce(b, preReduceOp, reduceOp, size, 0, 1, result);
	}
	/**
	 * Performs a reduce operation on the whole buffer.
	 * Calls {@code reduce(b, preReduceOp, reduceOp, b.getSize()/getElementSize(), 0, 1, result)}.
	 * @param b the input buffer
	 * @param preReduceOp an operation applied before the reduce operation is performed
	 * @param reduceOp the reduce operation
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 */
	public ReduceResult reduce(Buffer b, PreReduceOp preReduceOp, ReduceOp reduceOp,
			ReduceResult result) {
		return reduce(b, preReduceOp, reduceOp, b.getSize()/elementSize, result);
	}
	
	/**
	 * Merges two buffers into one and performs a reduction on the result.
	 * This is used to realise operations like the dot product, see 
	 * {@link #dotProduct(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.ReduceResult) }.
	 * @param a the first input buffer
	 * @param b the second input buffer
	 * @param mergeOp the operation to merge the two input buffers
	 * @param reduceOp the reduce operation
	 * @param size the count of elements to process
	 * @param offsetA the offset into the first input buffer
	 * @param offsetB the offset into the second input buffer
	 * @param stepA the step size of the first input buffer
	 * @param stepB the step size of the second input buffer
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 */
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
					a, b, new Kernel.LocalMemPerElement(elementSize), (int) size, tmpMem, (int) offsetA, (int) stepA, (int) offsetB, (int) stepB);
                clCommandQueue.finish();
		
		size = numWorkGroups;
		while (size > 1) {
			getReduceWorkSize((int) size, sizes);
			workGroupSize = sizes[0];
			numWorkGroups = sizes[1];
			globalWorkSize = numWorkGroups * workGroupSize;
			
			kernelOp2.Run2NoEvent(clCommandQueue, new Kernel.WorkSize(globalWorkSize), new Kernel.WorkSize(workGroupSize), 
						tmpMem, new Kernel.LocalMemPerElement(elementSize), (int) size, tmpMem, (int) 0, (int) 1);
                        clCommandQueue.finish();
			
			size = numWorkGroups;
		}
		
		result.event = tmpMem.copyToAsync(clCommandQueue, result.result, elementSize).register();
                clCommandQueue.finish();
		return result;
	}
	/**
	 * Merges two buffers into one and performs a reduction on the result.
	 * Calls {@code reduce2(a, b, mergeOp, reduceOp, size, 0, 0, 1, 1, result}.
	 * @param a the first input buffer
	 * @param b the second input buffer
	 * @param mergeOp the operation to merge the two input buffers
	 * @param reduceOp the reduce operation
	 * @param size the count of elements to process
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 * @see #reduce2(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MergeOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 */
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			ReduceOp reduceOp, long size, ReduceResult result) {
		return reduce2(a, b, mergeOp, reduceOp, size, 0, 0, 1, 1, result);
	}
	/**
	 * Merges two buffers into one and performs a reduction on the result.
	 * @param a the first input buffer
	 * @param b the second input buffer
	 * @param mergeOp the operation to merge the two input buffers
	 * @param reduceOp the reduce operation
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 * @see #reduce2(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MergeOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 */
	public ReduceResult reduce2(Buffer a, Buffer b, MergeOp mergeOp,
			ReduceOp reduceOp,	ReduceResult result) {
		long size = Math.min(a.getSize(), b.getSize()) / elementSize;
		return reduce2(a, b, mergeOp, reduceOp, size, result);
	}
	
	/**
	 * Computes the dot product of the two specified input buffers.
	 * This is a convenient method and simply calls 
	 * {@code reduce2(a, b, MergeOp.MUL, ReduceOp.ADD, result)}.
	 * @param a the first input buffer
	 * @param b the second input buffer
	 * @param result a result structure for reuse, {@code null} if a new one should be created
	 * @return the result
	 * @see #reduce2(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MergeOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 */
	public ReduceResult dotProduct(Buffer a, Buffer b, ReduceResult result) {
		return reduce2(a, b, MergeOp.MUL, ReduceOp.ADD, result);
	}
	
	/**
	 * Retrieves the result of a reduce operation in a blocking fashion.
	 * @param result the result structure from one of the reduce operations
	 * @return the resulting value
	 * @see #reduce(com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.PreReduceOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 * @see #reduce2(com.jme3.opencl.Buffer, com.jme3.opencl.Buffer, org.shaman.jmecl.utils.CLBlas.MergeOp, org.shaman.jmecl.utils.CLBlas.ReduceOp, long, long, long, long, long, org.shaman.jmecl.utils.CLBlas.ReduceResult) 
	 */
	@SuppressWarnings("unchecked")
	public T getReduceResultBlocking(ReduceResult result) {
		ByteBuffer buf = result.result.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		Number n = getter.get(buf);
		result.result.unmap(clCommandQueue, buf);
		return (T) n;
	}
}

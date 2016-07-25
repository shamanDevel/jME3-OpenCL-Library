/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.sorting;

import com.jme3.asset.AssetManager;
import com.jme3.math.FastMath;
import com.jme3.opencl.*;;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;

/**
 * Bitonic sort.
 * This is a comparison-based, in-place, non-stable sorting algorithm.
 * For simplicity and speed, the length of the input array must be a power of two.
 * 
 * <p>
 * Peters, H. , Schulz-Hildebrandt, O., Luttenberger, N.,
"Fast in-place, comparison-based sorting with cuda: a
study with bitonic sort," Concurrency and computation,
vol. 2011 (23), no. 7, pp. 681-693.
* http://onlinelibrary.wiley.com/store/10.1002/cpe.1686/asset/1686 ftp.
pdf?v=1&t=ioo1otg2&s=ad0c0650d7248a8d9d943cd0327bbcd7db093c0d
 * @author shaman
 */
public class BitonicSort implements Sorter {
	private static final Logger LOG = Logger.getLogger(BitonicSort.class.getName());
	private static final String PROGRAM_FILE = "org/shaman/jmecl/sorting/BitonicSort.cl";

	private final Context clContext;
	private final Device clDevice;
	private final CommandQueue clQueue;
	private final Sorter.ComparisonSettings settings;
	
	private int sharedMemorySize;
	private int workGroupSize;
	private Program program;
	private Kernel bitonicTrivialKernel;
	private Kernel bitonicSharedKernel;
	
	private final boolean useSharedMemory;

	/**
	 * Creates a new instance of the bitonic sort algorithm.
	 * @param openCLSettings the OpenCL settings
	 * @param comparisonSettings the comparison settings
	 */
	public BitonicSort(OpenCLSettings openCLSettings, Sorter.ComparisonSettings comparisonSettings) {
		this.clContext = openCLSettings.getClContext();
		this.clDevice = openCLSettings.getClCommandQueue().getDevice();
		this.clQueue = openCLSettings.getClCommandQueue();
		this.settings = comparisonSettings;
		this.useSharedMemory = true;
		
		String programHash = BitonicSort.class.getName() + "_" + settings.keyType
				+ "_" + settings.valueType + "_" + settings.comparison.hashCode();
		program = openCLSettings.getProgramCache().loadFromCache(programHash);
		if (program == null) {
			StringBuilder includes = new StringBuilder();
			includes.append("#define KEY_TYPE ").append(settings.keyType).append('\n');
			includes.append("#define VALUE_TYPE ").append(settings.valueType).append('\n');
			includes.append("#define COMPARISON_GREATER(x, y) ").append(settings.comparison).append('\n');
			program = clContext.createProgramFromSourceFilesWithInclude(openCLSettings.getAssetManager(), includes.toString(), PROGRAM_FILE);
			openCLSettings.getProgramCache().saveToCache(programHash, program);
		}
		program.build();
		bitonicTrivialKernel = program.createKernel("BitonicTrivial");
		bitonicSharedKernel = program.createKernel("BitonicShared");
		sharedMemorySize = (int) clDevice.getLocalMemorySize();
		workGroupSize = (int) clDevice.getMaxiumWorkItemsPerGroup();
	}

	@Override
	public boolean isComparisonBased() {
		return true;
	}

	@Override
	public boolean requiresPowerOfTwo() {
		return true;
	}

	@Override
	public void sort(Buffer keys, Buffer values) {
		int n = (int) (keys.getSize() / settings.keySize);
		int k = Math.round(FastMath.log(n, 2));
		if (1<<k != n) {
			//it must be a power of two
			throw new IllegalArgumentException("array size must be a power of two, but is "+n);
		}
		
		//analyse when to use shared memory
		int numKeysInShared = sharedMemorySize / (settings.keySize+settings.valueSize) / 4;
		int stepsInShared;
		if (useSharedMemory) {
			stepsInShared = Math.min(
				(int) FastMath.log(numKeysInShared, 2),
				(int) FastMath.log(workGroupSize * 2, 2));
		} else {
			stepsInShared = 0;
		}
	
		Kernel.WorkSize ws = new Kernel.WorkSize(n/2);
		for (int phase = 1; phase <= k; ++phase) {
			for (int step = phase; step > stepsInShared; --step) {
				bitonicTrivialKernel.Run1NoEvent(clQueue, ws, keys, values, phase, step);
			}
			//sort remaining ones in shared memory
			if (stepsInShared == 0) {
				continue;
			}
			int stepsLeft = Math.min(phase, stepsInShared);
			int threads = n/2;
			while (threads > workGroupSize) {
				threads /= 2;
			}
			Kernel.WorkSize wsl = new Kernel.WorkSize(threads);
			Kernel.WorkSize wsg = new Kernel.WorkSize(n/2);
			bitonicSharedKernel.Run2NoEvent(clQueue, wsg, wsl, keys, values, phase, stepsLeft, 
					new Kernel.LocalMem(threads * settings.keySize * 2), new Kernel.LocalMem(threads * settings.valueSize * 2));
		}
		
	}
	
	/**
	 * Testing only
	 * @param keys
	 * @param values 
	 */
	public void sortCPU(int[] keys, int[] values) {
		int n = keys.length;
		int k = Math.round(FastMath.log(n, 2));
		assert (1<<k == n); //it must be a power of two
		
		int numKeysInShared = sharedMemorySize / (settings.keySize+settings.valueSize) / 4;
		int stepsInShared = Math.min(
				(int) FastMath.log(numKeysInShared, 2),
				(int) FastMath.log(workGroupSize * 2, 2));
		

		Kernel.WorkSize ws = new Kernel.WorkSize(n/2);
		for (int phase = 1; phase <= k; ++phase) {
			for (int step = phase; step > stepsInShared; --step) {
				for (int idx = 0; idx < ws.getSizes()[0]; ++idx) {
					BitonicTrivial_deg1(keys, values, phase, step, idx);
				}
			}
			
			//sort remaining ones in shared memory
			int stepsLeft = Math.min(phase, stepsInShared);
			int threads = n/2;
			int workPerThread = 1;
			while (threads > workGroupSize) {
				threads /= 2;
			}
			Kernel.WorkSize wsl = new Kernel.WorkSize(threads);
			Kernel.WorkSize wsg = new Kernel.WorkSize(n/2);
			bitonicSharedKernelCPU(clQueue, wsg, wsl, keys, values, phase, stepsLeft, workPerThread, 
					new Kernel.LocalMem(threads * workPerThread * 2 * settings.keySize), new Kernel.LocalMem(threads * workPerThread * 2 * settings.valueSize));
		}
	}
	private void compareAndSwap(int[] keys, int[] values, int i, int j)
	{
		int k1 = keys[i];
		int k2 = keys[j];
		if (!(k1<=k2)) {
			keys[j] = k1;
			keys[i] = k2;
			int tmp = values[i];
			values[i] = values[j];
			values[j] = tmp;
		}
	}

	private void BitonicTrivial_deg1(int[] keys, int[] values, int phase, int step, int idx)
	{
		//unnormalized bitonic network
		int stepSize = 1 << (step-1);
		int part = idx / stepSize;
		int start = (part * 2 * stepSize) + (idx % stepSize);
		int direction = (idx / (1 << (phase-1))) % 2;
		int i,j;
		if (direction == 0) {
			i = start;
			j = start + stepSize;
		} else {
			j = start;
			i = start + stepSize;
		}
		compareAndSwap(keys, values, i, j);
	}
	
	private void BitonicTrivial_shared(
			int[] keys, int[] values, int phase, int step_, int workPerThread, int[] sharedKeys, int[] sharedValues,
			int global_id, int local_id, int local_size, int group_id, CyclicBarrier barrier) throws InterruptedException, BrokenBarrierException
	{
		int idx = global_id * workPerThread;
		int idxl = local_id * workPerThread;
		int offset = group_id * local_size * 2 * workPerThread;

		//1. load it into shared memory
		for (int k=0; k<workPerThread; ++k) {
			sharedKeys[idxl + k] = keys[idxl + k + offset];
			sharedValues[idxl + k] = values[idxl + k + offset];
		}
		for (int k=0; k<workPerThread; ++k) {
			sharedKeys[idxl + k + local_size] = keys[idxl + k + local_size + offset];
			sharedValues[idxl + k + local_size] = values[idxl + k + local_size + offset];
		}

		barrier.await();

		//2. sort locally		
		for (int step = step_; step>=1; --step) {
			for (int k=0; k<workPerThread; ++k) {
				int stepSize = 1 << (step-1);
				int part = idx / stepSize;
				int start = (part * 2 * stepSize) + (idx % stepSize);
				int direction = (idx / (1 << (phase-1))) % 2;
				int i,j;
				if (direction == 0) {
					i = start;
					j = start + stepSize;
				} else {
					j = start;
					i = start + stepSize;
				}
				compareAndSwap(sharedKeys, sharedValues, i-offset, j-offset);
			}
			barrier.await();
		}

		//3. write back
		for (int k=0; k<workPerThread; ++k) {
			keys[idxl + k + offset] = sharedKeys[idxl + k];
			values[idxl + k + offset] = sharedValues[idxl + k];
		}
		for (int k=0; k<workPerThread; ++k) {
			keys[idxl + k + local_size + offset] = sharedKeys[idxl + k + local_size];
			values[idxl + k + local_size + offset] = sharedValues[idxl + k + local_size];
		}
	}

	private void bitonicSharedKernelCPU(CommandQueue clQueue, Kernel.WorkSize wsg, Kernel.WorkSize wsl, 
			final int[] keys, final int[] values, final int phase, final int stepsLeft, final int workPerThread, Kernel.LocalMem keysMem, Kernel.LocalMem valuesMem) {
		int numBlocks = (int) (wsg.getSizes()[0] / wsl.getSizes()[0]);
		for (int b = 0; b < numBlocks; ++b) {
			final int fb = b;
			final int numThreads = (int) wsl.getSizes()[0];
			final CyclicBarrier barrier = new CyclicBarrier(numThreads);
			final int[] sharedKeys = new int[keysMem.getSize() / 4];
			final int[] sharedValues = new int[valuesMem.getSize() / 4];
			
			Thread[] threads = new Thread[numThreads];
			for (int i=0; i<numThreads; ++i) {
				final int fi = i;
				threads[i] = new Thread() {

					@Override
					public void run() {
						try {
							BitonicTrivial_shared(keys, values, phase, stepsLeft, workPerThread, sharedKeys, sharedValues, fb*numThreads+fi, fi, numThreads, fb, barrier);
						} catch (InterruptedException | BrokenBarrierException ex) {
							LOG.log(Level.SEVERE, null, ex);
						}
					}
					
				};
				threads[i].start();
			}
			for (int i=0; i<numThreads; ++i) {
				try {
					threads[i].join();
				} catch (InterruptedException ex) {
					LOG.log(Level.SEVERE, null, ex);
				}
			}
		}
	}
}

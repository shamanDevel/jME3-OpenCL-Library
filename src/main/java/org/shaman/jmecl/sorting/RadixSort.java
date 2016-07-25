/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.sorting;

import com.jme3.asset.AssetManager;
import com.jme3.opencl.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.shaman.jmecl.OpenCLSettings;

/**
 * Radix sort implementation.
 * Only integer keys and values are supported.
 * This is a stable, in-place, non-comparison-based algorithm.
 * You have to specify the range of bits to be sorted by {@link #setEndBit(int) }
 * (the start bit is assumed to be zero).
 * 
 * <p>
 * A. G. DUANE MERRILL, "High performance and scalable radix sorting:
A case study of implementing dynamic parallelism for gpu computing,"
Parallel processing letters, vol. 2011 (21), no. 2, pp. 245 - 272.
* 
* Geist Software Labs, \libcl: Parallel algorithm library."
* http://www.libcl.org
 * @author shaman
 */
public class RadixSort implements Sorter {
	private static final Logger LOG = Logger.getLogger(BitonicSort.class.getName());
	private static final String PROGRAM_FILE = "org/shaman/jmecl/sorting/RadixSort.cl";

	private final Context clContext;
	private final Device clDevice;
	private final CommandQueue clQueue;
	private final Sorter.ComparisonSettings settings;
	
	private final int RS_CBITS = 4;
	private final int RS_BLOCK_SIZE = 256;
	private final int RS_BLOCK_SIZE_CUBE = RS_BLOCK_SIZE*RS_BLOCK_SIZE*RS_BLOCK_SIZE;
	private final long cMaxArraySize = RS_BLOCK_SIZE_CUBE * 4 / (1 << RS_CBITS);
	
	private Program program;
	private final Kernel clBlockSort;
	private final Kernel clBlockScan;
	private final Kernel clBlockPrefix;
	private final Kernel clReorder;
	private Buffer bfTempKey;
	private Buffer bfTempVal;
	private Buffer bfBlockScan;
	private Buffer bfBlockSum;
	private Buffer bfBlockOffset;
	private int endBit = 32;
	
	/**
	 * Creates a new instance of the radix sort using ascending sorting on
	 * integer keys and values.
	 *
	 * @param settings the opencl settings
	 */
	public RadixSort(OpenCLSettings settings) {
		this.clContext = settings.getClContext();
		this.clDevice = settings.getClCommandQueue().getDevice();
		this.clQueue = settings.getClCommandQueue();
		this.settings = ComparisonSettings.INT_ASC;
		
		assert (this.settings.keyType.equals("int"));
		assert (this.settings.valueType.equals("int"));
		
		String programHash = RadixSort.class.getName();
		program = settings.getProgramCache().loadFromCache(programHash);
		if (program == null) {
			program = clContext.createProgramFromSourceFiles(settings.getAssetManager(), PROGRAM_FILE);
			settings.getProgramCache().saveToCache(programHash, program);
		}
		program.build();
		clBlockSort = program.createKernel("clBlockSort");
		clBlockScan = program.createKernel("clBlockScan");
		clBlockPrefix = program.createKernel("clBlockPrefix");
		clReorder = program.createKernel("clReorder");
	}

	private Buffer fit(Buffer buf, long iElements)
	{
		if (buf==null || buf.getSize()< iElements * 4) {
			if (buf != null) {
				buf.release();
			}
			return clContext.createBuffer(iElements * 4);
		}
		else {
			return buf;
		}
	}

	@Override
	public boolean isComparisonBased() {
		return false;
	}

	@Override
	public boolean requiresPowerOfTwo() {
		return false;
	}
 
	public void setEndBit(int endBit) {
		this.endBit = endBit;
	}
	
	@Override
	public void sort(Buffer keys, Buffer values) {
		int n = (int) (keys.getSize() / 4);
		
		radixsort(keys, values, 0, endBit, n);
	}
	
	private void radixsort(Buffer bfKey, Buffer bfVal, int iStartBit, int iEndBit, long elements)
	{
		if (bfKey.getSize()>= cMaxArraySize * 4)
		{
			LOG.log(Level.SEVERE, "maximum sortable array size = {0}", cMaxArraySize);
			return;
		}

		if ((iEndBit - iStartBit) % RS_CBITS != 0)
		{
			LOG.log(Level.SEVERE, "end bit({0}) - start bit({1}) must be divisible by 4", new Object[]{iEndBit, iStartBit});
			return;
		}

		int lBlockCount = (int) Math.ceil((float)elements / (RS_BLOCK_SIZE));
		bfBlockScan = fit(bfBlockScan, lBlockCount*(1 << RS_CBITS));
		bfBlockOffset = fit(bfBlockOffset, lBlockCount*(1 << RS_CBITS));
		bfBlockSum = fit(bfBlockSum, RS_BLOCK_SIZE);

		int lElementCount = (int) elements;
		bfTempKey = fit(bfTempKey, lElementCount);
		bfTempVal = fit(bfTempVal, lElementCount);

		int lGlobalSize = lBlockCount*RS_BLOCK_SIZE;
		int lScanCount = lBlockCount*(1 << RS_CBITS) / 4;
		int lScanSize = (int) (Math.ceil((float)lScanCount / RS_BLOCK_SIZE)*RS_BLOCK_SIZE);

		Kernel.WorkSize lws = new Kernel.WorkSize(RS_BLOCK_SIZE);
		for (int j = iStartBit; j<iEndBit; j += RS_CBITS)
		{
			clBlockSort.Run2NoEvent(clQueue, new Kernel.WorkSize(lGlobalSize), lws,
				bfKey, bfTempKey, bfVal, bfTempVal, j, bfBlockScan, bfBlockOffset, lElementCount);

			clBlockScan.Run2NoEvent(clQueue, new Kernel.WorkSize(lScanSize), lws,
				bfBlockScan, bfBlockSum, lScanCount);

			clBlockPrefix.Run2NoEvent(clQueue, new Kernel.WorkSize(lScanSize), lws,
				bfBlockScan, bfBlockSum, lScanCount);

			clReorder.Run2NoEvent(clQueue, new Kernel.WorkSize(lGlobalSize), lws,
				bfTempKey, bfKey, bfTempVal, bfVal, bfBlockScan, bfBlockOffset, j, lElementCount);
		}
	}
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl;

import com.jme3.asset.AssetManager;
import com.jme3.opencl.*;
import com.jme3.util.BufferUtils;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.shaman.jmecl.sorting.BitonicSort;
import org.shaman.jmecl.sorting.RadixSort;
import org.shaman.jmecl.sorting.Sorter;

import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Weiss
 */
public class SortingTest extends AbstractOpenCLTest {
	private static final Logger LOG = Logger.getLogger(SortingTest.class.getName());
	
	private static int[] keyOriginal;
	private static Buffer keyBuffer;
	private static int[] valueOriginal;
	private static Buffer valueBuffer;
	
	public SortingTest() {
	}
	
	
	
	@Test
	public void testBitonicSort() {
		int sizes[] = {1<<5, 1<<7, 1<<9, 1<<13, 1<<18};
		BitonicSort sort = new BitonicSort(settings, Sorter.ComparisonSettings.INT_ASC);
		for (int size : sizes) {
			testBitonicSort(sort, size);
		}
	}
	private void testBitonicSort(BitonicSort sort, int length) {
		createKeysValues(length);
		sort.sort(keyBuffer, valueBuffer);
		checkSorted(length);
		releaseBuffers();
	}
	
	@Test
	public void testRadixSort() {
		int radix[] = {4, 8, 12};
		int sizes[] = {1<<6, 5618, 1<<15, 47201, 34879};
		RadixSort sort = new RadixSort(settings);
		for (int r : radix) {
			for (int size : sizes) {
				testRadixSort(sort, size, r);
			}
		}
	}
	private void testRadixSort(RadixSort sort, int length, int maxBit) {
		System.out.println("Test Radix Sort, length="+length+" maxBit="+maxBit);
		createKeysValues(length, 1<<maxBit);
		sort.setEndBit(maxBit);
		sort.sort(keyBuffer, valueBuffer);
		checkSortedStable(length);
		releaseBuffers();
	}
	
	private void createKeysValues(int size) {
		createKeysValues(size, size);
	}
	private void createKeysValues(int size, int maxValue) {
		//create keys and values
		System.out.println("create keys and values");
		keyOriginal = new int[size];
		valueOriginal = new int[size];
		for (int i=0; i<size; ++i) {
			int key = rand.nextInt(maxValue);
			int value = i+1;
			keyOriginal[i] = key;
			valueOriginal[i] = value;
		}
		System.out.println("keys and values created");
		//create buffers
		System.out.println("create buffers");
		keyBuffer = clContext.createBuffer(size * 4);
		valueBuffer = clContext.createBuffer(size * 4);
		ByteBuffer bb = BufferUtils.createByteBuffer(size * 4);
		bb.asIntBuffer().put(keyOriginal);
		keyBuffer.write(clCommandQueue, bb);
		bb.asIntBuffer().put(valueOriginal);
		valueBuffer.write(clCommandQueue, bb);
		System.out.println("buffers created");
	}
	
	private static void releaseBuffers() {
		keyBuffer.release();
		valueBuffer.release();
		System.out.println("buffers released");
	}
	
	private static void checkSorted(int size, int[] keys, int[] values) {
		//1. keys must be sorted
		for (int i=1; i<size; ++i) {
			if (!(keys[i-1] <= keys[i])) {
				fail("not ordered at index "+(i-1)+","+i+": "+keys[i-1]+"<="+keys[i]+" does not hold");
			}
		}
	}
	
	private void checkSorted(int size) {
		int[] keys = new int[size];
		int[] values = new int[size];
		ByteBuffer bb = BufferUtils.createByteBuffer(size * 4);
		keyBuffer.read(clCommandQueue, bb);
		bb.asIntBuffer().get(keys);
		valueBuffer.read(clCommandQueue, bb);
		bb.asIntBuffer().get(values);
		
		checkSorted(size, keys, values);
	}
	
	private void checkSortedStable(int size) {
		//sort original keys
		SortingItem[] items = new SortingItem[size];
		for (int i=0; i<size; ++i) {
			items[i] = new SortingItem(keyOriginal[i], valueOriginal[i]);
		}
		Arrays.sort(items);
		//query result
		ByteBuffer keyBuf = keyBuffer.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		ByteBuffer valueBuf = valueBuffer.map(clCommandQueue, MappingAccess.MAP_READ_ONLY);
		keyBuf.rewind();
		valueBuf.rewind();
		//compare with gpu sort
		for (int i=0; i<size; ++i) {
			assertEquals("keys do not match at index "+i, items[i].key, keyBuf.getInt());
			assertEquals("values do not match at index "+i, items[i].value, valueBuf.getInt());
		}
		keyBuf.rewind();
		valueBuf.rewind();
		keyBuffer.unmap(clCommandQueue, keyBuf);
		valueBuffer.unmap(clCommandQueue, valueBuf);
	}
	private static class SortingItem implements Comparable<SortingItem> {
		private int key, value;
		public SortingItem(int key, int value) {
			this.key = key;
			this.value = value;
		}
		@Override
		public int compareTo(SortingItem o) {
			return Integer.compare(key, o.key);
		}
	}
}

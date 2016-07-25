/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.sorting;

import com.jme3.opencl.Buffer;

/**
 * Interface for the various sorting algorithms.
 * Some algorithms may work on arbitrary types with a user-defined comparison
 * function. That can be specified by the {@link ComparisonSettings} class
 * @author shaman
 */
public interface Sorter {

	/**
	 * Checks if the sorting algorithm is comparison based.
	 * @return {@code true} iff it is comparison based
	 */
	boolean isComparisonBased();
	
	/**
	 * Checks if the sorting algorithm requires the input size to be a power of two.
	 * @return {@code true} iff the length of the sorted sequence must be a power of two
	 */
	boolean requiresPowerOfTwo();
	
	/**
	 * Sorts the given array of (key, value) pairs.
	 * The sorting happens in-place
	 * @param keys the key buffer
	 * @param values the value buffer
	 */
	void sort(Buffer keys, Buffer values);
	
	/**
	 * Specifies settings of the used datatypes.
	 * Instances of this class are passed to the constructor of the sorting 
	 * algorithms that support them.
	 */
	public static class ComparisonSettings {
		public final int keySize;
		public final String keyType;
		public final int valueSize;
		public final String valueType;
		public final String comparison;

		/**
		 * Constucts a new instance of the settings
		 * @param keySize the size of a key in bytes
		 * @param keyType the OpenCL type of a key
		 * @param valueSize the size of a value in bytes
		 * @param valueType the OpenCL type of a value
		 * @param comparison the comparison function. The two values are available
		 * as {@code x} and {@code y}. Example: {@code (x) <= (y)}
		 */
		public ComparisonSettings(int keySize, String keyType, int valueSize, String valueType, String comparison) {
			this.keySize = keySize;
			this.keyType = keyType;
			this.valueSize = valueSize;
			this.valueType = valueType;
			this.comparison = comparison;
		}
		
		/**
		 * Settings for sorting integer keys and values in ascending order.
		 */
		public static ComparisonSettings INT_ASC = new Sorter.ComparisonSettings(4, "int", 4, "int", "(x) <= (y)");
		/**
		 * Settings for sorting integer keys and values in descending order.
		 */
		public static ComparisonSettings INT_DESC = new Sorter.ComparisonSettings(4, "int", 4, "int", "(x) >= (y)");
	}
}

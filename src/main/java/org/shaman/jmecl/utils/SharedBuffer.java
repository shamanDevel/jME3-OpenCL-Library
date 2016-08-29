/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.opencl.Buffer;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.opencl.MemoryAccess;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.opengl.GLRenderer;
import com.jme3.scene.VertexBuffer;
import com.jme3.util.BufferUtils;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Represents a vertex buffer that can be access by both jME rendering system
 * via {@link VertexBuffer} and by OpenCL via {@link Buffer}.
 * @author Sebastian Weiss
 */
public class SharedBuffer {
	private static final Logger LOG = Logger.getLogger(SharedBuffer.class.getName());
	/**
	 * If save mode is enabled, all modifications work an a separate buffer
	 * and are copied once per frame to the vertex buffer.
	 * This is a workaround, because sometimes the real vertex buffer is cleared
	 * and all the information is lost when it is resized.
	 */
	private static final boolean SAVE_MODE = true;
	
	private VertexBuffer jmeBuffer;
	private Buffer clBuffer;
	private Buffer clBuffer2;
	private GLRenderer renderer;
	private Context clContext;
	
	private VertexBuffer.Format format;
	private VertexBuffer.Type type;
	private int size;
	private int components;
	private MemoryAccess ma;

	/**
	 * Creates a new vertex buffer with the specified properties and shares it 
	 * with OpenCL. The initial buffer will be uninitialized.
	 * @param format the format of the buffer
	 * @param type the type of each component
	 * @param components the number of components per entry
	 * @param size the size of the buffer
	 */
	public SharedBuffer(VertexBuffer.Format format, VertexBuffer.Type type, int components, int size) {
		this.format = format;
		this.type = type;
		this.components = components;
		this.size = size;
	}

	/**
	 * Creates a shared buffer from the specified vertex buffer
	 * @param vertexBuffer the vertex buffer to use
	 */
	public SharedBuffer(VertexBuffer vertexBuffer) {
		this.jmeBuffer = vertexBuffer;
		this.format = vertexBuffer.getFormat();
		this.type = vertexBuffer.getBufferType();
		this.components = vertexBuffer.getNumComponents();
		this.size = vertexBuffer.getNumElements();
	}
	
	public boolean isInitialized() {
		return clBuffer != null;
	}
	
	private java.nio.Buffer createBuffer(int size) {
		switch (format) {
			case Byte:
			case UnsignedByte:
				return BufferUtils.createByteBuffer(size);
			case Double: return BufferUtils.createDoubleBuffer(size);
			case Float: return BufferUtils.createFloatBuffer(size);
			case Half:
			case Short:
			case UnsignedShort:
				return BufferUtils.createShortBuffer(size);
			case Int:
			case UnsignedInt:
				return BufferUtils.createIntBuffer(size);
			default:
				throw new IllegalArgumentException("unknown format "+format);
		}
	}
	
	/**
	 * Initializes the shared buffer.
	 * The buffer is created (not with the constructor taking the vertex buffer as argument),
	 * uploaded to the GPU and shared with OpenCL.
	 * Must be called from the jME main thread.
	 * 
	 * @param renderManager the render manager
	 * @param clContext the OpenCL context
	 * @param memoryAccess the allowed memory access to the OpenCL texture
	 * 
	 * @throws IllegalStateException if it was already initialized
	 * @throws IllegalArgumentException if the underlying renderer implementation is not {@code GLRenderer}
	 */
	public void initialize(RenderManager renderManager, Context clContext, MemoryAccess memoryAccess) {
		if (isInitialized()) {
			throw new IllegalStateException("already initialized");
		}
		//get renderer
		if (renderManager.getRenderer() instanceof GLRenderer) {
			renderer = (GLRenderer) renderManager.getRenderer();
		} else {
			throw new IllegalArgumentException("Only GLRenderer supported");
		}
		
		//create buffer
		if (jmeBuffer == null) {
			jmeBuffer = new VertexBuffer(type);
			java.nio.Buffer data = createBuffer(size * components);
			jmeBuffer.setupData(VertexBuffer.Usage.Dynamic, components, format, data);
			LOG.info("vertex buffer created");
		}
		
		//upload to gpu
		renderer.updateBufferData(jmeBuffer);
		LOG.log(Level.INFO, "uploaded to the GPU: {0}", jmeBuffer.getId());
		
		//create shared bufer
		clBuffer2 = clContext.bindVertexBuffer(jmeBuffer, memoryAccess).register();
		if (SAVE_MODE) {
			clBuffer = clContext.createBuffer(size * components * format.getComponentSize(), memoryAccess).register();
		} else {
			clBuffer = clBuffer2;
		}
		LOG.log(Level.FINE, "OpenCL buffer created from vertex buffer: {0}", clBuffer);
		
		this.clContext = clContext;
		this.ma = memoryAccess;
	}

	public VertexBuffer getJMEBuffer() {
		return jmeBuffer;
	}

	public Buffer getCLBuffer() {
		return clBuffer;
	}
	
	public void aquireCLBuffer(CommandQueue commandQueue) {
		if (!SAVE_MODE) {
			clBuffer.acquireBufferForSharingNoEvent(commandQueue);
		}
	}
	
	public void releaseCLBuffer(CommandQueue commandQueue) {
		if (SAVE_MODE) {
			//copy to vertex buffer
			clBuffer2.acquireBufferForSharingNoEvent(commandQueue);
			long size1 = clBuffer.getSize();
			long size2 = clBuffer2.getSize();
			if (size1 > size2) {
				LOG.severe("wrong sizes: clBuffer.getSize()="+size1+", clBuffer2.getSize()="+size2);
			}
			clBuffer.copyToAsync(commandQueue, clBuffer2, Math.min(size1, size2)).release();
			clBuffer2.releaseBufferForSharingNoEvent(commandQueue);
		} else {
			clBuffer.releaseBufferForSharingNoEvent(commandQueue);
		}
	}
	
	/**
	 * Resizes this buffer (shrink and grow).
	 * Must be called from the jME main thread.
	 * 
	 * The buffer entries that are appended are uninitialized, the old ones are copied,
	 * but only if a command queue is passed. If no command queue is passed
	 * (parameter is set to {@code null}, nothing is passed)
	 * 
	 * @param newSize the new size
	 * @param queue the command queue. Must be not null of the old data should be
	 * copied
	 */
	public void resize(int newSize, CommandQueue queue) {
		if (size == newSize) {
			return;
		}
		LOG.log(Level.INFO, "resizing shared buffer from {0} to {1}", new Object[]{size, newSize});
		size = newSize;
		
		Buffer oldCLBuffer = clBuffer;
		Buffer oldCLBuffer2 = clBuffer2;
		
		//create new vertex buffer
		jmeBuffer = new VertexBuffer(type);
		int bufferSize = size * components;
		java.nio.Buffer newData = createBuffer(bufferSize);
		
		jmeBuffer.setupData(VertexBuffer.Usage.Dynamic, components, format, newData);
		
		//upload to gpu
		renderer.updateBufferData(jmeBuffer);
		
		//create shared bufer
		clBuffer2 = clContext.bindVertexBuffer(jmeBuffer, ma).register();
		if (SAVE_MODE) {
			clBuffer = clContext.createBuffer(size * components * format.getComponentSize(), ma).register();
		} else {
			clBuffer = clBuffer2;
		}
		
		//copy old to new
		if (queue != null) {
			if (SAVE_MODE) {
				oldCLBuffer.copyToAsync(queue, clBuffer, Math.min(oldCLBuffer.getSize(), clBuffer.getSize())).release();
			} else {
				clBuffer.acquireBufferForSharingNoEvent(queue);
				oldCLBuffer.acquireBufferForSharingNoEvent(queue);
				oldCLBuffer.copyToAsync(queue, clBuffer, Math.min(oldCLBuffer.getSize(), clBuffer.getSize())).release();
				clBuffer.releaseBufferForSharingNoEvent(queue);
				oldCLBuffer.releaseBufferForSharingNoEvent(queue);
			}
		}
		
		//delete old
		oldCLBuffer.release();
		if (SAVE_MODE) {
			oldCLBuffer2.release();
		}
	}
}

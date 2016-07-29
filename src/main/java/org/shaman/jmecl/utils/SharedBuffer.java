/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.opencl.Buffer;
import com.jme3.opencl.Context;
import com.jme3.opencl.MemoryAccess;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.opengl.GLRenderer;
import com.jme3.scene.VertexBuffer;
import com.jme3.util.BufferUtils;
import java.nio.ByteBuffer;

/**
 * Represents a vertex buffer that can be access by both jME rendering system
 * via {@link VertexBuffer} and by OpenCL via {@link Buffer}.
 * @author Sebastian Weiss
 */
public class SharedBuffer {
	private VertexBuffer jmeBuffer;
	private Buffer clBuffer;
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
			ByteBuffer data = BufferUtils.createByteBuffer(size * components * format.getComponentSize());
			jmeBuffer.setupData(VertexBuffer.Usage.Dynamic, components, format, data);
		}
		
		//upload to gpu
		renderer.updateBufferData(jmeBuffer);
		
		//create shared bufer
		clBuffer = clContext.bindVertexBuffer(jmeBuffer, memoryAccess).register();
		
		this.clContext = clContext;
		this.ma = memoryAccess;
	}

	public VertexBuffer getJMEBuffer() {
		return jmeBuffer;
	}

	public Buffer getCLBuffer() {
		return clBuffer;
	}
	
	/**
	 * Resizes this buffer (shrink and grow).
	 * Must be called from the jME main thread.
	 * 
	 * The buffer entries that are appended are uninitialized, the old ones are copied.
	 * 
	 * @param newSize the new size
	 */
	public void resize(int newSize) {
		if (size == newSize) {
			return;
		}
		
		//delete old buffer
		clBuffer.release();
		
		//create new data buffer and copy values
		ByteBuffer newData = BufferUtils.createByteBuffer(size * components * format.getComponentSize());
		ByteBuffer oldData = (ByteBuffer) jmeBuffer.getData();
		oldData.rewind();
		oldData.limit(Math.min(oldData.limit(), newData.remaining()));
		newData.put(oldData);
		newData.rewind();
		
		//create new vertex buffer
		jmeBuffer = new VertexBuffer(type);
		jmeBuffer.setupData(VertexBuffer.Usage.Dynamic, components, format, newData);
		
		//upload to gpu
		renderer.updateBufferData(jmeBuffer);
		
		//create shared bufer
		clBuffer = clContext.bindVertexBuffer(jmeBuffer, ma).register();
	}
}

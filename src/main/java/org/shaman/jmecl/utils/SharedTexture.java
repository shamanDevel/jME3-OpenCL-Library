/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.opencl.Context;
import com.jme3.opencl.MemoryAccess;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.opengl.GLRenderer;
import com.jme3.texture.*;
import com.jme3.texture.image.ColorSpace;
import java.util.ArrayList;

/**
 * This class represents a texture that can be access by the 
 * jME3 rendering system (via the {@link Texture} class) and by opencl
 * (via {@link com.jme3.opencl.Image}).
 * 
 * Note that prior to usage, it has to be initialized.
 * 
 * TODO: not cloneable and not saveable (probably shouldn't as well
 * 
 * @author shaman
 */
public class SharedTexture {
	
	private Texture texture;
	private Image image;
	private com.jme3.opencl.Image clImage;
	private com.jme3.opencl.Image.ImageDescriptor descriptor;
	private Image.Format format;
	private GLRenderer renderer;

	/**
	 * Creates a new shared texture with the specified dimensions, type and format
	 * @param descriptor specifies dimensions and type
	 * @param format specifies the format
	 */
	public SharedTexture(com.jme3.opencl.Image.ImageDescriptor descriptor, Image.Format format) {
		if (descriptor == null) {
			throw new NullPointerException("descriptor is null");
		}
		if (format == null) {
			throw new NullPointerException("format is null");
		}
		this.descriptor = descriptor;
		this.format = format;
	}
	
	/**
	 * Creates a new shared texture from an already existing jME texture.
	 * Only the creation of the shared OpenCL image is left
	 * @param texture 
	 */
	public SharedTexture(Texture texture) {
		this.texture = texture;
		this.image = texture.getImage();
	}

	/**
	 * Checks if the image is already initialized.
	 * @return {@code true} if already initialized
	 */
	public boolean isInitialized() {
		return clImage != null;
	}
	
	/**
	 * Initializes the shared textures.
	 * The texture is created (not with the constructor taking the texture as argument),
	 * uploaded to the GPU and shared with OpenCL.
	 * Must be called from the jME main thread.
	 * 
	 * @param renderManager the render manager
	 * @param clContext the OpenCL context
	 * @param memoryAccess the allowed memory access to the OpenCL texture
	 * 
	 * @throws IllegalStateException if it was already initialized
	 * @throws IllegalArgumentException 
	 *   If the underlying renderer implementation is not {@code GLRenderer}, or
	 *   if the OpenCL texture type can't be mapped to a texture type of jME
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
		
		//create texture
		if (texture == null) {
			switch (descriptor.type) {
				case IMAGE_2D:
					texture = new Texture2D((int) descriptor.width, (int) descriptor.height, format);
					break;
				case IMAGE_2D_ARRAY:
					ArrayList<Image> images = new ArrayList<>((int) descriptor.arraySize);
					for (int i=0; i<descriptor.arraySize; ++i) {
						images.add(new Image(format, (int) descriptor.width, (int) descriptor.height, null, ColorSpace.Linear));
					}
					texture = new TextureArray(images);
					break;
				case IMAGE_3D:
					texture = new Texture3D((int) descriptor.width, (int) descriptor.height, (int) descriptor.depth, format);
					break;
				default:
					throw new IllegalArgumentException("unsupported texture type: "+descriptor.type);
			}
			image = texture.getImage();
		}
		
		//upload texture to GPU
		renderer.updateTexImageData(image, texture.getType(), 0, false);
		
		//create shared image
		clImage = clContext.bindImage(texture, memoryAccess).register();
	}
	
	/**
	 * Returns the jME texture.
	 * @return 
	 */
	public Texture getJMETexture() {
		return texture;
	}
	
	/**
	 * Returns the OpenCL image
	 * @return 
	 */
	public com.jme3.opencl.Image getCLImage() {
		return clImage;
	}
}

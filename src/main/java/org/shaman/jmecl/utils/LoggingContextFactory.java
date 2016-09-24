/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.asset.AssetManager;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Matrix3f;
import com.jme3.math.Matrix4f;
import com.jme3.math.Quaternion;
import com.jme3.math.Vector2f;
import com.jme3.math.Vector4f;
import com.jme3.opencl.Buffer;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.opencl.Device;
import com.jme3.opencl.Event;
import com.jme3.opencl.Image;
import com.jme3.opencl.Kernel;
import com.jme3.opencl.KernelCompilationException;
import com.jme3.opencl.MappingAccess;
import com.jme3.opencl.MemoryAccess;
import com.jme3.opencl.OpenCLObject;
import com.jme3.opencl.Program;
import com.jme3.scene.VertexBuffer;
import com.jme3.texture.FrameBuffer;
import com.jme3.texture.Texture;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Creates a wrapper for a {@link Context} that issues a {@code clFinish()}
 * after every GPU call and does other additional checks.
 * @author Sebastian
 */
public class LoggingContextFactory {
	/**
	 * The logger, everything is reported over it.
	 */
	private static final Logger LOG = Logger.getLogger(LoggingContextFactory.class.getName());
	
	/**
	 * Kernel calls are logged.
	 */
	public static final int FLAG_LOG_KERNEL = 1;
	/**
	 * Memory operations are logged.
	 */
	public static final int FLAG_LOG_MEMOP = 2;
	/**
	 * Acquiring+releasing of shared resources are logged.
	 */
	public static final int FLAG_LOG_ACQUIRE = 4;
	/**
	 * Object creation is logged.
	 */
	public static final int FLAG_LOG_CREATION = 8;
	/**
	 * Object deletion is logged.
	 */
	public static final int FLAG_LOG_DELETION = 16;
	/**
	 * Registering an object for automatic garbage collection is logged.
	 */
	public static final int FLAG_LOG_REGISTER = 32;
	/**
	 * Logs calls to clFinish() and clFlush().
	 */
	public static final int FLAG_LOG_FINISH = 64;
	/**
	 * Logs event functions.
	 */
	public static final int FLAG_LOG_EVENT = 128;
	
	public static final int FLAG_LOG_ALL = 0xffffffff;
	
	public static Context createLoggingContext(Context delegate, int flags) {
		return new LoggingContext(delegate, flags);
	}
	
	public static Context createLoggingContext(Context delegate) {
		return new LoggingContext(delegate, FLAG_LOG_ALL);
	}
	
	private static void throwWrongType(Object obj) {
		throw new IllegalArgumentException("argument "+obj+" ("+obj.getClass()+") was not created by the longging wrapper");
	}
	
	private static class LoggingReleaser implements OpenCLObject.ObjectReleaser {
		private final OpenCLObject.ObjectReleaser delegate;
		private final String objName;

		public LoggingReleaser(String objName, OpenCLObject.ObjectReleaser delegate) {
			this.objName = objName;
			this.delegate = delegate;
		}
		
		@Override
		public void release() {
			if (objName != null) LOG.log(Level.INFO, "release {0}", objName);
			delegate.release();
		}
		
	}
	
	private static class LoggingContext extends Context {
		private final Context delegate;
		private final int flags;

		public LoggingContext(Context delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()));
			this.delegate = delegate;
			this.flags = flags;
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.info("DebugContext created");
		}

		@Override
		public Context register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public List<? extends Device> getDevices() {
			return delegate.getDevices();
		}

		@Override
		public CommandQueue createQueue() {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create command queue with the default device", this);
			return new LoggingCommandQueue(delegate.createQueue(), flags);
		}

		@Override
		public CommandQueue createQueue(Device device) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create command queue with {1}", new Object[]{this, device});
			return new LoggingCommandQueue(delegate.createQueue(device), flags);
		}

		@Override
		public Buffer createBuffer(long l, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create buffer of size {1}B and memory access {2}", new Object[]{this, l, ma});
			return new LoggingBuffer(delegate.createBuffer(l, ma), flags);
		}

		@Override
		public Buffer createBuffer(long l) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create buffer of size {1}B and default memory access", new Object[]{this, l});
			return new LoggingBuffer(delegate.createBuffer(l), flags);
		}

		@Override
		public Buffer createBufferFromHost(ByteBuffer bb, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create buffer from host {1} and memory access {2}", new Object[]{this, bb, ma});
			return new LoggingBuffer(delegate.createBufferFromHost(bb, ma), flags);
		}

		@Override
		public Buffer createBufferFromHost(ByteBuffer bb) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create buffer from host {1} and default memory access", new Object[]{this, bb});
			return new LoggingBuffer(delegate.createBufferFromHost(bb), flags);
		}

		@Override
		public Image createImage(MemoryAccess ma, Image.ImageFormat i, Image.ImageDescriptor id) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create image with format {1}, descriptor {2} and memory access {3}", new Object[]{this, i, id, ma});
			return new LoggingImage(delegate.createImage(ma, i, id), flags);
		}

		@Override
		public Image.ImageFormat[] querySupportedFormats(MemoryAccess ma, Image.ImageType it) {
			return delegate.querySupportedFormats(ma, it);
		}

		@Override
		public Buffer bindVertexBuffer(VertexBuffer vb, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: bind vertex buffer {1} with memory access {2}", new Object[]{this, vb, ma});
			return new LoggingBuffer(delegate.bindVertexBuffer(vb, ma), flags);
		}

		@Override
		public Image bindImage(com.jme3.texture.Image image, Texture.Type type, int i, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: bind image {1} with texture type {2} mipmap level {3} and mem access {4}", new Object[]{this, image, type, i, ma});
			return new LoggingImage(delegate.bindImage(image, type, i, ma), flags);
		}

		@Override
		public Image bindImage(Texture txtr, int i, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: bind texture {1}, mipmap level {2}, memory access {3}", new Object[]{this, txtr, i, ma});
			return new LoggingImage(delegate.bindImage(txtr, i, ma), flags);
		}

		@Override
		public Image bindImage(Texture txtr, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: bind texture {1} with memory access {2}", new Object[]{this, txtr, ma});
			return new LoggingImage(delegate.bindImage(txtr, ma), flags);
		}

		@Override
		public Image bindRenderBuffer(FrameBuffer.RenderBuffer rb, MemoryAccess ma) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: bind render buffer {1} with  memory access {2}", new Object[]{this, rb, ma});
			return new LoggingImage(delegate.bindRenderBuffer(rb, ma), flags);
		}

		@Override
		public Program createProgramFromSourceCode(String string) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create program with source code\n{1}", new Object[]{this, string});
			return new LoggingProgram(delegate.createProgramFromSourceCode(string), flags);
		}

		@Override
		public Program createProgramFromSourceCodeWithDependencies(String string, AssetManager am) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create program with source code and dependencies\n{1}", new Object[]{this, string});
			return super.createProgramFromSourceCodeWithDependencies(string, am);
		}

		@Override
		public Program createProgramFromSourceFilesWithInclude(AssetManager am, String string, List<String> list) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "{0}: create program with source files {2} and include \"{1}\"", new Object[]{this, string, list});
			return super.createProgramFromSourceFilesWithInclude(am, string, list);
		}

		@Override
		public Program createProgramFromBinary(ByteBuffer bb, Device device) {
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.info(this+": create program from binary "+bb+" for device "+device);
			return new LoggingProgram(delegate.createProgramFromBinary(bb, device), flags);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		protected Image bindPureRenderBuffer(FrameBuffer.RenderBuffer rb, MemoryAccess ma) {
			throw new UnsupportedOperationException("Not supported yet.");
		}
		
	}
	
	private static class LoggingCommandQueue extends CommandQueue {
		private final CommandQueue delegate;
		private final int flags;

		public LoggingCommandQueue(CommandQueue delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()), null);
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public CommandQueue register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public Device getDevice() {
			return delegate.getDevice();
		}

		@Override
		public void flush() {
			if ((flags & FLAG_LOG_FINISH) != 0) LOG.log(Level.INFO, "{0}: flush()", this);
			delegate.flush();
		}

		@Override
		public void finish() {
			if ((flags & FLAG_LOG_FINISH) != 0) LOG.log(Level.INFO, "{0}: finish()", this);
			delegate.finish();
		}

		@Override
		public int hashCode() {
			return delegate.hashCode();
		}

		@Override
		public boolean equals(Object o) {
			return delegate.equals(o);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}
		
	}
	
	private static class LoggingEvent extends Event {
		private final Event delegate;
		private final int flags;

		public LoggingEvent(Event delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()));
			this.delegate = delegate;
			this.flags = flags;
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.info("event created");
		}

		@Override
		public Event register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public void waitForFinished() {
			if ((flags & FLAG_LOG_EVENT) != 0) LOG.log(Level.INFO, "{0}: waitForFinished()", this);
			delegate.waitForFinished();
			if ((flags & FLAG_LOG_EVENT) != 0) LOG.log(Level.INFO, "{0}: waiting done", this);
		}

		@Override
		public boolean isCompleted() {
			boolean b = delegate.isCompleted();
			if ((flags & FLAG_LOG_EVENT) != 0) LOG.log(Level.INFO, "{0}: isCompleted={1}", new Object[]{this, b});
			return b;
		}

		@Override
		public int hashCode() {
			return delegate.hashCode();
		}

		@Override
		public boolean equals(Object o) {
			return delegate.equals(o);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}
		
	}
	
	private static class LoggingBuffer extends Buffer {
		private final Buffer delegate;
		private final int flags;

		public LoggingBuffer(Buffer delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()));
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public Buffer register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public long getSize() {
			return delegate.getSize();
		}

		@Override
		public MemoryAccess getMemoryAccessFlags() {
			return delegate.getMemoryAccessFlags();
		}

		@Override
		public void read(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: read from {1}, size={2}, offset={3}", new Object[]{this, bb, l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.read(((LoggingCommandQueue) cq).delegate, bb, l, l1);
		}

		@Override
		public void read(CommandQueue cq, ByteBuffer bb, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: read from {1}, size={2}", new Object[]{this, bb, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.read(((LoggingCommandQueue) cq).delegate, bb, l);
		}

		@Override
		public void read(CommandQueue cq, ByteBuffer bb) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: read from {1}", new Object[]{this, bb});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.read(((LoggingCommandQueue) cq).delegate, bb);
		}

		@Override
		public Event readAsync(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async read from {1}, size={2}, offset={3}", new Object[]{this, bb, l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.readAsync(((LoggingCommandQueue) cq).delegate, bb, l, l1), flags);
		}

		@Override
		public Event readAsync(CommandQueue cq, ByteBuffer bb, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async read from {1}, size={2}", new Object[]{this, bb, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.readAsync(((LoggingCommandQueue) cq).delegate, bb, l), flags);
		}

		@Override
		public Event readAsync(CommandQueue cq, ByteBuffer bb) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async read from {1}", new Object[]{this, bb});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.readAsync(((LoggingCommandQueue) cq).delegate, bb), flags);
		}

		@Override
		public void write(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: write to {1}, size={2}, offset={3}", new Object[]{this, bb, l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.write(((LoggingCommandQueue) cq).delegate, bb, l, l1);
		}

		@Override
		public void write(CommandQueue cq, ByteBuffer bb, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: write to {1}, size={2}", new Object[]{this, bb, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.write(((LoggingCommandQueue) cq).delegate, bb, l);
		}

		@Override
		public void write(CommandQueue cq, ByteBuffer bb) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: write to {1}", new Object[]{this, bb});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.write(((LoggingCommandQueue) cq).delegate, bb);
		}

		@Override
		public Event writeAsync(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async write to {1}, size={2}, offset={3}", new Object[]{this, bb, l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.writeAsync(((LoggingCommandQueue) cq).delegate, bb, l, l1), flags);
		}

		@Override
		public Event writeAsync(CommandQueue cq, ByteBuffer bb, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async write to {1}, size={2}", new Object[]{this, bb, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.writeAsync(((LoggingCommandQueue) cq).delegate, bb, l), flags);
		}

		@Override
		public Event writeAsync(CommandQueue cq, ByteBuffer bb) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async write to {1}", new Object[]{this, bb});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.writeAsync(((LoggingCommandQueue) cq).delegate, bb), flags);
		}

		@Override
		public void copyTo(CommandQueue cq, Buffer buffer, long l, long l1, long l2) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "copy {0} to {1}, size={2}, src offset={3}, dest offset={4}", new Object[]{this, buffer, l, l1, l2});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			delegate.copyTo(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate, l, l1, l2);
		}

		@Override
		public void copyTo(CommandQueue cq, Buffer buffer, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "copy {0} to {1}, size={2}", new Object[]{this, buffer, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			delegate.copyTo(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate, l);
		}

		@Override
		public void copyTo(CommandQueue cq, Buffer buffer) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "copy {0} to {1}", new Object[]{this, buffer});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			delegate.copyTo(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate);
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Buffer buffer, long l, long l1, long l2) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "async copy {0} to {1}, size={2}, src offset={3}, dest offset={4}", new Object[]{this, buffer, l, l1, l2});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			return new LoggingEvent(delegate.copyToAsync(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate, l, l1, l2), flags);
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Buffer buffer, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "async copy {0} to {1}, size={2}", new Object[]{this, buffer, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			return new LoggingEvent(delegate.copyToAsync(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate, l), flags);
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Buffer buffer) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "async copy {0} to {1}", new Object[]{this, buffer});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			return new LoggingEvent(delegate.copyToAsync(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate), flags);
		}

		@Override
		public ByteBuffer map(CommandQueue cq, long l, long l1, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: map with access {1}, size={2}, offset={3}", new Object[]{this, ma, l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.map(((LoggingCommandQueue) cq).delegate, l, l1, ma);
		}

		@Override
		public ByteBuffer map(CommandQueue cq, long l, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: map with access {1}, size={2}", new Object[]{this, ma, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.map(((LoggingCommandQueue) cq).delegate, l, ma);
		}

		@Override
		public ByteBuffer map(CommandQueue cq, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: map with access {1}", new Object[]{this, ma});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.map(((LoggingCommandQueue) cq).delegate, ma);
		}

		@Override
		public void unmap(CommandQueue cq, ByteBuffer bb) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: unmap {1}", new Object[]{this, bb});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.unmap(((LoggingCommandQueue) cq).delegate, bb);
		}

		@Override
		public AsyncMapping mapAsync(CommandQueue cq, long l, long l1, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async map with access {1}, size={2}, offset={3}", new Object[]{this, ma, l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.mapAsync(((LoggingCommandQueue) cq).delegate, l, l1, ma);
		}

		@Override
		public AsyncMapping mapAsync(CommandQueue cq, long l, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async map with access {1}, size={2}", new Object[]{this, ma, l});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.mapAsync(((LoggingCommandQueue) cq).delegate, l, ma);
		}

		@Override
		public AsyncMapping mapAsync(CommandQueue cq, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async map with access {1}", new Object[]{this, ma});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.mapAsync(((LoggingCommandQueue) cq).delegate, ma);
		}

		@Override
		public Event fillAsync(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.info(this+": fill with pattern "+bb+", size="+l+", offset="+l1);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.fillAsync(((LoggingCommandQueue) cq).delegate, bb, l, l1), flags);
		}

		@Override
		public Event copyToImageAsync(CommandQueue cq, Image image, long l, long[] longs, long[] longs1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "copy {0} to {1}, src offset={2}, dest origin={3}, dest region={4}", new Object[]{this, image, l, Arrays.toString(longs), Arrays.toString(longs1)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(image instanceof LoggingImage)) throwWrongType(image);
			return new LoggingEvent(delegate.copyToImageAsync(((LoggingCommandQueue) cq).delegate, ((LoggingImage) image).delegate, l, longs, longs1), flags);
		}

		@Override
		public Event acquireBufferForSharingAsync(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "aquire {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.acquireBufferForSharingAsync(((LoggingCommandQueue) cq).delegate), flags);
		}

		@Override
		public void acquireBufferForSharingNoEvent(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "aquire {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.acquireBufferForSharingNoEvent(((LoggingCommandQueue) cq).delegate);
		}

		@Override
		public Event releaseBufferForSharingAsync(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "release {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.releaseBufferForSharingAsync(((LoggingCommandQueue) cq).delegate), flags);
		}

		@Override
		public void releaseBufferForSharingNoEvent(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "release {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.releaseBufferForSharingNoEvent(((LoggingCommandQueue) cq).delegate);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public int hashCode() {
			return delegate.hashCode();
		}

		@Override
		public boolean equals(Object o) {
			return delegate.equals(o);
		}
		
		
	}
	
	private static class LoggingImage extends Image {
		private final Image delegate;
		private final int flags;

		public LoggingImage(Image delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()));
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public Image register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public long getWidth() {
			return delegate.getWidth();
		}

		@Override
		public long getHeight() {
			return delegate.getHeight();
		}

		@Override
		public long getDepth() {
			return delegate.getDepth();
		}

		@Override
		public long getRowPitch() {
			return delegate.getRowPitch();
		}

		@Override
		public long getSlicePitch() {
			return delegate.getSlicePitch();
		}

		@Override
		public long getArraySize() {
			return delegate.getArraySize();
		}

		@Override
		public ImageFormat getImageFormat() {
			return delegate.getImageFormat();
		}

		@Override
		public ImageType getImageType() {
			return delegate.getImageType();
		}

		@Override
		public int getElementSize() {
			return delegate.getElementSize();
		}

		@Override
		public void readImage(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: read from {1}, origin={2}, region={3}, row offset={4}, slice offset={5}", new Object[]{this, bb, Arrays.toString(longs), Arrays.toString(longs1), l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.readImage(((LoggingCommandQueue) cq).delegate, bb, longs, longs1, l, l1);
		}

		@Override
		public Event readImageAsync(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async read from {1}, origin={2}, region={3}, row offset={4}, slice offset={5}", new Object[]{this, bb, Arrays.toString(longs), Arrays.toString(longs1), l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.readImageAsync(((LoggingCommandQueue) cq).delegate, bb, longs, longs1, l, l1), flags);
		}

		@Override
		public void writeImage(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: write to {1}, origin={2}, region={3}, row offset={4}, slice offset={5}", new Object[]{this, bb, Arrays.toString(longs), Arrays.toString(longs1), l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.writeImage(((LoggingCommandQueue) cq).delegate, bb, longs, longs1, l, l1);
		}

		@Override
		public Event writeImageAsync(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async write to {1}, origin={2}, region={3}, row offset={4}, slice offset={5}", new Object[]{this, bb, Arrays.toString(longs), Arrays.toString(longs1), l, l1});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.writeImageAsync(((LoggingCommandQueue) cq).delegate, bb, longs, longs1, l, l1), flags);
		}

		@Override
		public void copyTo(CommandQueue cq, Image image, long[] longs, long[] longs1, long[] longs2) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "copy {0} to {1}, src origin={2}, dest origin={3}, region={4}", new Object[]{this, image, Arrays.toString(longs), Arrays.toString(longs1), Arrays.toString(longs2)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(image instanceof LoggingImage)) throwWrongType(image);
			delegate.copyTo(((LoggingCommandQueue) cq).delegate, ((LoggingImage) image).delegate, longs, longs1, longs2);
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Image image, long[] longs, long[] longs1, long[] longs2) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "async copy {0} to {1}, src origin={2}, dest origin={3}, region={4}", new Object[]{this, image, Arrays.toString(longs), Arrays.toString(longs1), Arrays.toString(longs2)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(image instanceof LoggingImage)) throwWrongType(image);
			return new LoggingEvent(delegate.copyToAsync(((LoggingCommandQueue) cq).delegate, ((LoggingImage) image).delegate, longs, longs1, longs2), flags);
		}

		@Override
		public ImageMapping map(CommandQueue cq, long[] longs, long[] longs1, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: map with memory access {1}, origin={2}, region={3}", new Object[]{this, ma, Arrays.toString(longs), Arrays.toString(longs1)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.map(((LoggingCommandQueue) cq).delegate, longs, longs1, ma);
		}

		@Override
		public ImageMapping mapAsync(CommandQueue cq, long[] longs, long[] longs1, MappingAccess ma) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: async map with memory access {1}, origin={2}, region={3}", new Object[]{this, ma, Arrays.toString(longs), Arrays.toString(longs1)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return delegate.mapAsync(((LoggingCommandQueue) cq).delegate, longs, longs1, ma);
		}

		@Override
		public void unmap(CommandQueue cq, ImageMapping im) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: unmap {1}", new Object[]{this, im.buffer});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.unmap(((LoggingCommandQueue) cq).delegate, im);
		}

		@Override
		public Event fillAsync(CommandQueue cq, long[] longs, long[] longs1, ColorRGBA crgba) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: fill with {1}, origin={2}, region={3}", new Object[]{this, crgba, Arrays.toString(longs), Arrays.toString(longs1)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.fillAsync(((LoggingCommandQueue) cq).delegate, longs, longs1, crgba), flags);
		}

		@Override
		public Event fillAsync(CommandQueue cq, long[] longs, long[] longs1, int[] ints) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "{0}: fill with {1}, origin={2}, region={3}", new Object[]{this, Arrays.toString(ints), Arrays.toString(longs), Arrays.toString(longs1)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.fillAsync(((LoggingCommandQueue) cq).delegate, longs, longs1, ints), flags);
		}

		@Override
		public Event copyToBufferAsync(CommandQueue cq, Buffer buffer, long[] longs, long[] longs1, long l) {
			if ((flags & FLAG_LOG_MEMOP) != 0) LOG.log(Level.INFO, "async copy {0} to {1}, src origin={2}, region={4}, dest offset={3}", new Object[]{this, buffer, Arrays.toString(longs), l, Arrays.toString(longs1)});
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			return new LoggingEvent(delegate.copyToBufferAsync(((LoggingCommandQueue) cq).delegate, ((LoggingBuffer) buffer).delegate, longs, longs1, l), flags);
		}

		@Override
		public Event acquireImageForSharingAsync(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "aquire {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.acquireImageForSharingAsync(((LoggingCommandQueue) cq).delegate), flags);
		}

		@Override
		public void acquireImageForSharingNoEvent(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "aquire {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.acquireImageForSharingNoEvent(((LoggingCommandQueue) cq).delegate);
		}

		@Override
		public Event releaseImageForSharingAsync(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "release {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.releaseImageForSharingAsync(((LoggingCommandQueue) cq).delegate), flags);
		}

		@Override
		public void releaseImageForSharingNoEvent(CommandQueue cq) {
			if ((flags & FLAG_LOG_ACQUIRE) != 0) LOG.log(Level.INFO, "release {0}", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.releaseImageForSharingNoEvent(((LoggingCommandQueue) cq).delegate);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public int hashCode() {
			return delegate.hashCode();
		}

		@Override
		public boolean equals(Object o) {
			return delegate.equals(o);
		}
			
	}
	
	private static class LoggingProgram extends Program {
		private final Program delegate;
		private final int flags;

		public LoggingProgram(Program delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()));
			this.delegate = delegate;
			this.flags = flags;
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.log(Level.INFO, "program {0} created", this);
		}

		@Override
		public Program register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public void build(String string, Device... devices) throws KernelCompilationException {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: build with args {1} on the devices {2}", new Object[]{this, string, Arrays.toString(devices)});
			delegate.build(string, devices);
		}

		@Override
		public void build() throws KernelCompilationException {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: build", this);
			delegate.build();
		}

		@Override
		public Kernel createKernel(String string) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: create kernel with the name {1}", new Object[]{this, string});
			return new LoggingKernel(delegate.createKernel(string), flags);
		}

		@Override
		public Kernel[] createAllKernels() {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: create all kernels", this);
			Kernel[] kx = delegate.createAllKernels();
			Kernel[] dkx = new Kernel[kx.length];
			for (int i=0; i<kx.length; ++i) {
				dkx[i] = new LoggingKernel(kx[i], flags);
			}
			return dkx;
		}

		@Override
		public ByteBuffer getBinary(Device device) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: get binary for device {1}", new Object[]{this, device});
			return delegate.getBinary(device);
		}

		@Override
		public int hashCode() {
			return delegate.hashCode();
		}

		@Override
		public boolean equals(Object o) {
			return delegate.equals(o);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}
		
	}
	
	private static class LoggingKernel extends Kernel {
		private final Kernel delegate;
		private final int flags;

		public LoggingKernel(Kernel delegate, int flags) {
			super(new LoggingReleaser(((flags & FLAG_LOG_DELETION) != 0) ? delegate.toString() : null, delegate.getReleaser()));
			this.delegate = delegate;
			this.flags = flags;
			if ((flags & FLAG_LOG_CREATION) != 0) LOG.info("kernel "+this+" created");
		}

		@Override
		public Kernel register() {
			delegate.register();
			if ((flags & FLAG_LOG_REGISTER)!=0) LOG.log(Level.INFO, "{0} registered", this);
			return this;
		}

		@Override
		public String getName() {
			return delegate.getName();
		}

		@Override
		public int getArgCount() {
			return delegate.getArgCount();
		}

		@Override
		public WorkSize getGlobalWorkSize() {
			return delegate.getGlobalWorkSize();
		}

		@Override
		public void setGlobalWorkSize(WorkSize ws) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set global work size to {1}", new Object[]{this, ws});
			delegate.setGlobalWorkSize(ws);
		}

		@Override
		public void setGlobalWorkSize(int i) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set global work size to {1}", new Object[]{this, i});
			delegate.setGlobalWorkSize(i);
		}

		@Override
		public void setGlobalWorkSize(int i, int i1) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set global work size to ({1}, {2})", new Object[]{this, i, i1});
			delegate.setGlobalWorkSize(i, i1);
		}

		@Override
		public void setGlobalWorkSize(int i, int i1, int i2) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set global work size to ({1}, {2}, {3})", new Object[]{this, i, i1, i2});
			delegate.setGlobalWorkSize(i, i1, i2);
		}

		@Override
		public WorkSize getWorkGroupSize() {
			return delegate.getWorkGroupSize();
		}

		@Override
		public void setWorkGroupSize(WorkSize ws) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set work group size to {1}", new Object[]{this, ws});
			delegate.setWorkGroupSize(ws);
		}

		@Override
		public void setWorkGroupSize(int i) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set work group size to {1}", new Object[]{this, i});
			delegate.setWorkGroupSize(i);
		}

		@Override
		public void setWorkGroupSize(int i, int i1) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set work group size to ({1}, {2})", new Object[]{this, i, i1});
			delegate.setWorkGroupSize(i, i1);
		}

		@Override
		public void setWorkGroupSdize(int i, int i1, int i2) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set work group size to ({1}, {2}, {3})", new Object[]{this, i, i1, i2});
			delegate.setWorkGroupSdize(i, i1, i2);
		}

		@Override
		public void setWorkGroupSizeToNull() {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set work group size to null", this);
			delegate.setWorkGroupSizeToNull();
		}

		@Override
		public long getMaxWorkGroupSize(Device device) {
			return delegate.getMaxWorkGroupSize(device);
		}

		@Override
		public void setArg(int i, LocalMemPerElement lmpe) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, lmpe});
			delegate.setArg(i, lmpe);
		}

		@Override
		public void setArg(int i, LocalMem lm) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, lm});
			delegate.setArg(i, lm);
		}

		@Override
		public void setArg(int i, Buffer buffer) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, buffer});
			if (!(buffer instanceof LoggingBuffer)) throwWrongType(buffer);
			delegate.setArg(i, ((LoggingBuffer) buffer).delegate);
		}

		@Override
		public void setArg(int i, Image image) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, image});
			if (!(image instanceof LoggingImage)) throwWrongType(image);
			delegate.setArg(i, ((LoggingImage) image).delegate);
		}

		@Override
		public void setArg(int i, byte b) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to (byte) {2}", new Object[]{this, i, b});
			delegate.setArg(i, b);
		}

		@Override
		public void setArg(int i, short s) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to (short) {2}", new Object[]{this, i, s});
			delegate.setArg(i, s);
		}

		@Override
		public void setArg(int i, int i1) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to (int) {2}", new Object[]{this, i, i1});
			delegate.setArg(i, i1);
		}

		@Override
		public void setArg(int i, long l) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to (long) {2}", new Object[]{this, i, l});
			delegate.setArg(i, l);
		}

		@Override
		public void setArg(int i, float f) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to (float) {2}", new Object[]{this, i, f});
			delegate.setArg(i, f);
		}

		@Override
		public void setArg(int i, double d) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to (double) {2}", new Object[]{this, i, d});
			delegate.setArg(i, d);
		}

		@Override
		public void setArg(int i, Vector2f vctrf) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, vctrf});
			delegate.setArg(i, vctrf);
		}

		@Override
		public void setArg(int i, Vector4f vctrf) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, vctrf});
			delegate.setArg(i, vctrf);
		}

		@Override
		public void setArg(int i, Quaternion qtrn) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, qtrn});
			delegate.setArg(i, qtrn);
		}

		@Override
		public void setArg(int i, Matrix4f mtrxf) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, mtrxf});
			delegate.setArg(i, mtrxf);
		}

		@Override
		public void setArg(int i, Matrix3f mtrxf) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set arg {1} to {2}", new Object[]{this, i, mtrxf});
			delegate.setArg(i, mtrxf);
		}

		@Override
		public void setArg(int i, ByteBuffer bb, long l) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: set raw arg {1} to {2} with size ", new Object[]{this, i, bb, l});
			delegate.setArg(i, bb, l);
		}

		@Override
		public Event Run(CommandQueue cq) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: direct call", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.Run(((LoggingCommandQueue) cq).delegate), flags);
		}

		@Override
		public void RunNoEvent(CommandQueue cq) {
			if ((flags & FLAG_LOG_KERNEL) != 0) LOG.log(Level.INFO, "{0}: direct call", this);
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.RunNoEvent(((LoggingCommandQueue) cq).delegate);
		}
		
		private Object[] unwrap(Object[] os) {
			Object[] os2 = new Object[os.length];
			for (int i=0; i<os.length; ++i) {
				if (os[i] instanceof Buffer) {
					if (!(os[i] instanceof LoggingBuffer)) throwWrongType(os[i]);
					os2[i] = ((LoggingBuffer) os[i]).delegate;
				} else if (os[i] instanceof Image) {
					if (!(os[i] instanceof LoggingImage)) throwWrongType(os[i]);
					os2[i] = ((LoggingImage) os[i]).delegate;
				} else {
					os2[i] = os[i];
				}
			}
			return os2;
		}

		private Object[] unwrap(Object[] os, StringBuilder args) {
			Object[] os2 = new Object[os.length];
			for (int i=0; i<os.length; ++i) {
				if (i>0) {
					args.append(", ");
				}
				if (os[i] instanceof Buffer) {
					if (!(os[i] instanceof LoggingBuffer)) throwWrongType(os[i]);
					os2[i] = ((LoggingBuffer) os[i]).delegate;
					args.append(os2[i]);
				} else if (os[i] instanceof Image) {
					if (!(os[i] instanceof LoggingImage)) throwWrongType(os[i]);
					os2[i] = ((LoggingImage) os[i]).delegate;
					args.append(os2[i]);
				} else if (os[i] instanceof Byte) {
					os2[i] = os[i];
					args.append("(byte) ").append(os[i]);
				} else if (os[i] instanceof Short) {
					os2[i] = os[i];
					args.append("(short) ").append(os[i]);
				} else if (os[i] instanceof Integer) {
					os2[i] = os[i];
					args.append("(int) ").append(os[i]);
				} else if (os[i] instanceof Long) {
					os2[i] = os[i];
					args.append("(long) ").append(os[i]);
				} else if (os[i] instanceof Float) {
					os2[i] = os[i];
					args.append("(float) ").append(os[i]);
				} else if (os[i] instanceof Double) {
					os2[i] = os[i];
					args.append("(double) ").append(os[i]);
				} else {
					os2[i] = os[i];
				}
			}
			return os2;
		}
		
		@Override
		public Event Run1(CommandQueue cq, WorkSize ws, Object... os) {
			if ((flags & FLAG_LOG_KERNEL) != 0) {
				StringBuilder s = new StringBuilder();
				s.append(getName()).append('(');
				os = unwrap(os, s);
				s.append(")  global work size=").append(ws);
				LOG.info(s.toString());
			} else {
				os = unwrap(os);
			}
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.Run1(((LoggingCommandQueue) cq).delegate, ws, os), flags);
		}

		@Override
		public void Run1NoEvent(CommandQueue cq, WorkSize ws, Object... os) {
			if ((flags & FLAG_LOG_KERNEL) != 0) {
				StringBuilder s = new StringBuilder();
				s.append(getName()).append('(');
				os = unwrap(os, s);
				s.append(")  global work size=").append(ws);
				LOG.info(s.toString());
			} else {
				os = unwrap(os);
			}
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.Run1NoEvent(((LoggingCommandQueue) cq).delegate, ws, os);
		}

		@Override
		public Event Run2(CommandQueue cq, WorkSize ws, WorkSize ws1, Object... os) {
			if ((flags & FLAG_LOG_KERNEL) != 0) {
				StringBuilder s = new StringBuilder();
				s.append(getName()).append('(');
				os = unwrap(os, s);
				s.append(")  global work size=").append(ws);
				s.append("  local work group size=").append(ws1);
				LOG.info(s.toString());
			} else {
				os = unwrap(os);
			}
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			return new LoggingEvent(delegate.Run2(((LoggingCommandQueue) cq).delegate, ws, ws1, os), flags);
		}

		@Override
		public void Run2NoEvent(CommandQueue cq, WorkSize ws, WorkSize ws1, Object... os) {
			if ((flags & FLAG_LOG_KERNEL) != 0) {
				StringBuilder s = new StringBuilder();
				s.append(getName()).append('(');
				os = unwrap(os, s);
				s.append(")  global work size=").append(ws);
				s.append("  local work group size=").append(ws1);
				LOG.info(s.toString());
			} else {
				os = unwrap(os);
			}
			if (!(cq instanceof LoggingCommandQueue)) throwWrongType(cq);
			delegate.Run2NoEvent(((LoggingCommandQueue) cq).delegate, ws, ws1, os);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public int hashCode() {
			return delegate.hashCode();
		}

		@Override
		public boolean equals(Object o) {
			return delegate.equals(o);
		}
		
	}
}

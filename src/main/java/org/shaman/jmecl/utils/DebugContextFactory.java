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
import com.jme3.opencl.Program;
import com.jme3.scene.VertexBuffer;
import com.jme3.texture.FrameBuffer;
import com.jme3.texture.Texture;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Creates a wrapper for a {@link Context} that issues a {@code clFinish()}
 * after every GPU call and does other additional checks.
 * @author Sebastian
 */
public class DebugContextFactory {
	
	/**
	 * clFinish is called after each kernel call.
	 */
	private static final int FLAG_FINISH_AFTER_KERNEL = 1;
	/**
	 * clFinish is called after each memory operation.
	 */
	private static final int FLAG_FINISH_AFTER_MEMOP = 2;
	/**
	 * clFinish is called after each mem/image acquiring/releasing.
	 */
	private static final int FLAG_FINISH_AFTER_ACQUIRE = 4;
	/**
	 * All tests are performed.
	 */
	private static final int FLAG_ALL = FLAG_FINISH_AFTER_KERNEL | FLAG_FINISH_AFTER_MEMOP | FLAG_FINISH_AFTER_ACQUIRE;

	private static final Logger LOG = Logger.getLogger(DebugContextFactory.class.getName());
	
	public static Context createDebugContext(Context delegate) {
		return createDebugContext(delegate, FLAG_ALL);
	}
	public static Context createDebugContext(Context delegate, int flags) {
		LOG.log(Level.INFO, "running debug context that delegates to {0}", delegate);
		return new DebugContext(delegate, flags);
	}
	
	private static void throwWrongType(Object obj) {
		throw new IllegalArgumentException("argument "+obj+" ("+obj.getClass()+") was not created by the debug wrapper");
	}
	
	private static class DebugContext extends Context {
		private final Context delegate;
		private final int flags;

		public DebugContext(Context delegate, int flags) {
			super(null);
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public Context register() {
			delegate.register();
			return this;
		}

		@Override
		public List<? extends Device> getDevices() {
			return delegate.getDevices();
		}

		@Override
		public CommandQueue createQueue() {
			return new DebugCommandQueue(delegate.createQueue(), flags);
		}

		@Override
		public CommandQueue createQueue(Device device) {
			return new DebugCommandQueue(delegate.createQueue(device), flags);
		}

		@Override
		public Buffer createBuffer(long l, MemoryAccess ma) {
			return new DebugBuffer(delegate.createBuffer(l, ma), flags, false);
		}

		@Override
		public Buffer createBuffer(long l) {
			return new DebugBuffer(delegate.createBuffer(l), flags, false);
		}

		@Override
		public Buffer createBufferFromHost(ByteBuffer bb, MemoryAccess ma) {
			return new DebugBuffer(delegate.createBufferFromHost(bb, ma), flags, false);
		}

		@Override
		public Buffer createBufferFromHost(ByteBuffer bb) {
			return new DebugBuffer(delegate.createBufferFromHost(bb), flags, false);
		}

		@Override
		public Image createImage(MemoryAccess ma, Image.ImageFormat i, Image.ImageDescriptor id) {
			return new DebugImage(delegate.createImage(ma, i, id), flags, false);
		}

		@Override
		public Image.ImageFormat[] querySupportedFormats(MemoryAccess ma, Image.ImageType it) {
			return delegate.querySupportedFormats(ma, it);
		}

		@Override
		public Buffer bindVertexBuffer(VertexBuffer vb, MemoryAccess ma) {
			return new DebugBuffer(delegate.bindVertexBuffer(vb, ma), flags, true);
		}

		@Override
		public Image bindImage(com.jme3.texture.Image image, Texture.Type type, int i, MemoryAccess ma) {
			return new DebugImage(delegate.bindImage(image, type, i, ma), flags, true);
		}

		@Override
		public Image bindImage(Texture txtr, int i, MemoryAccess ma) {
			return new DebugImage(delegate.bindImage(txtr, i, ma), flags, true);
		}

		@Override
		public Image bindImage(Texture txtr, MemoryAccess ma) {
			return new DebugImage(delegate.bindImage(txtr, ma), flags, true);
		}

		@Override
		public Image bindRenderBuffer(FrameBuffer.RenderBuffer rb, MemoryAccess ma) {
			return new DebugImage(delegate.bindRenderBuffer(rb, ma), flags, true);
		}

		@Override
		public Program createProgramFromSourceCode(String string) {
			return new DebugProgram(delegate.createProgramFromSourceCode(string), flags);
		}

		@Override
		public Program createProgramFromSourceCodeWithDependencies(String string, AssetManager am) {
			return new DebugProgram(delegate.createProgramFromSourceCodeWithDependencies(string, am), flags);
		}

		@Override
		public Program createProgramFromSourceFilesWithInclude(AssetManager am, String string, String... strings) {
			return new DebugProgram(delegate.createProgramFromSourceFilesWithInclude(am, string, strings), flags);
		}

		@Override
		public Program createProgramFromSourceFilesWithInclude(AssetManager am, String string, List<String> list) {
			return new DebugProgram(delegate.createProgramFromSourceFilesWithInclude(am, string, list), flags);
		}

		@Override
		public Program createProgramFromSourceFiles(AssetManager am, String... strings) {
			return new DebugProgram(delegate.createProgramFromSourceFiles(am, strings), flags);
		}

		@Override
		public Program createProgramFromSourceFiles(AssetManager am, List<String> list) {
			return new DebugProgram(delegate.createProgramFromSourceFiles(am, list), flags);
		}

		@Override
		public Program createProgramFromBinary(ByteBuffer bb, Device device) {
			return new DebugProgram(delegate.createProgramFromBinary(bb, device), flags);
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
		}

		@Override
		protected Image bindPureRenderBuffer(FrameBuffer.RenderBuffer rb, MemoryAccess ma) {
			throw new UnsupportedOperationException("Not supported yet.");
		}
		
	}
	
	private static class DebugCommandQueue extends CommandQueue {
		private final CommandQueue delegate;
		private final int flags;

		public DebugCommandQueue(CommandQueue delegate, int flags) {
			super(null, null);
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public CommandQueue register() {
			delegate.register();
			return this;
		}

		@Override
		public Device getDevice() {
			return delegate.getDevice();
		}

		@Override
		public void flush() {
			delegate.flush();
		}

		@Override
		public void finish() {
			delegate.finish();
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
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
	
	private static class DebugEvent extends Event {
		private final Event delegate;
		private final int flags;

		public DebugEvent(Event delegate, int flags) {
			super(null);
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public Event register() {
			delegate.register();
			return this;
		}

		@Override
		public void waitForFinished() {
			delegate.waitForFinished();
		}

		@Override
		public boolean isCompleted() {
			return delegate.isCompleted();
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
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
	
	private static class DebugBuffer extends Buffer {
		private final Buffer delegate;
		private final int flags;
		private final boolean shared;
		private boolean aquired;

		public DebugBuffer(Buffer delegate, int flags, boolean shared) {
			super(null);
			this.delegate = delegate;
			this.flags = flags;
			this.shared = shared;
		}

		private void checkAccess() {
			if (shared && !aquired) {
				throw new IllegalStateException("shared buffer is not aquired");
			}
		}
		
		@Override
		public Buffer register() {
			delegate.register();
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
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.read(((DebugCommandQueue) cq).delegate, bb, l, l1);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public void read(CommandQueue cq, ByteBuffer bb, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.read(((DebugCommandQueue) cq).delegate, bb, l);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public void read(CommandQueue cq, ByteBuffer bb) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.read(((DebugCommandQueue) cq).delegate, bb);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event readAsync(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.readAsync(((DebugCommandQueue) cq).delegate, bb, l, l1), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event readAsync(CommandQueue cq, ByteBuffer bb, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.readAsync(((DebugCommandQueue) cq).delegate, bb, l), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event readAsync(CommandQueue cq, ByteBuffer bb) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.readAsync(((DebugCommandQueue) cq).delegate, bb), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public void write(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.write(((DebugCommandQueue) cq).delegate, bb, l, l1);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public void write(CommandQueue cq, ByteBuffer bb, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.write(((DebugCommandQueue) cq).delegate, bb, l);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public void write(CommandQueue cq, ByteBuffer bb) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.write(((DebugCommandQueue) cq).delegate, bb);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event writeAsync(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.writeAsync(((DebugCommandQueue) cq).delegate, bb, l, l1), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event writeAsync(CommandQueue cq, ByteBuffer bb, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.writeAsync(((DebugCommandQueue) cq).delegate, bb, l), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event writeAsync(CommandQueue cq, ByteBuffer bb) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.writeAsync(((DebugCommandQueue) cq).delegate, bb), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public void copyTo(CommandQueue cq, Buffer buffer, long l, long l1, long l2) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			delegate.copyTo(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate, l, l1, l2);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public void copyTo(CommandQueue cq, Buffer buffer, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			delegate.copyTo(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate, l);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public void copyTo(CommandQueue cq, Buffer buffer) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			delegate.copyTo(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Buffer buffer, long l, long l1, long l2) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			Event e = new DebugEvent(delegate.copyToAsync(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate, l, l1, l2), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Buffer buffer, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			Event e = new DebugEvent(delegate.copyToAsync(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate, l), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Buffer buffer) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			Event e = new DebugEvent(delegate.copyToAsync(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public ByteBuffer map(CommandQueue cq, long l, long l1, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			ByteBuffer b = delegate.map(((DebugCommandQueue) cq).delegate, l, l1, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return b;
		}

		@Override
		public ByteBuffer map(CommandQueue cq, long l, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			ByteBuffer b = delegate.map(((DebugCommandQueue) cq).delegate, l, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return b;
		}

		@Override
		public ByteBuffer map(CommandQueue cq, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			ByteBuffer b = delegate.map(((DebugCommandQueue) cq).delegate, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return b;
		}

		@Override
		public void unmap(CommandQueue cq, ByteBuffer bb) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.unmap(((DebugCommandQueue) cq).delegate, bb);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public AsyncMapping mapAsync(CommandQueue cq, long l, long l1, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			AsyncMapping m = delegate.mapAsync(((DebugCommandQueue) cq).delegate, l, l1, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return m;
		}

		@Override
		public AsyncMapping mapAsync(CommandQueue cq, long l, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			AsyncMapping m = delegate.mapAsync(((DebugCommandQueue) cq).delegate, l, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return m;
		}

		@Override
		public AsyncMapping mapAsync(CommandQueue cq, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			AsyncMapping m = delegate.mapAsync(((DebugCommandQueue) cq).delegate, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return m;
		}

		@Override
		public Event fillAsync(CommandQueue cq, ByteBuffer bb, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.fillAsync(((DebugCommandQueue) cq).delegate, bb, l, l1), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event copyToImageAsync(CommandQueue cq, Image image, long l, long[] longs, long[] longs1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(image instanceof DebugImage)) throwWrongType(image);
			((DebugImage) image).checkAccess();
			Event e = new DebugEvent(delegate.copyToImageAsync(((DebugCommandQueue) cq).delegate, ((DebugImage) image).delegate, l, longs, longs1), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event acquireBufferForSharingAsync(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to acquire a buffer that is not shared");
			} else {
				if (aquired) {
					throw new IllegalStateException("buffer already acquired");
				}
			}
			aquired = true;
			Event e = new DebugEvent(delegate.acquireBufferForSharingAsync(((DebugCommandQueue) cq).delegate), flags);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
			return e;
		}

		@Override
		public void acquireBufferForSharingNoEvent(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to acquire a buffer that is not shared");
			} else {
				if (aquired) {
					throw new IllegalStateException("buffer already acquired");
				}
			}
			aquired = true;
			delegate.acquireBufferForSharingNoEvent(((DebugCommandQueue) cq).delegate);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
		}

		@Override
		public Event releaseBufferForSharingAsync(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to release a buffer that is not shared");
			} else {
				if (!aquired) {
					throw new IllegalStateException("buffer already released");
				}
			}
			aquired = false;
			Event e = new DebugEvent(delegate.releaseBufferForSharingAsync(((DebugCommandQueue) cq).delegate), flags);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
			return e;
		}

		@Override
		public void releaseBufferForSharingNoEvent(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to release a buffer that is not shared");
			} else {
				if (!aquired) {
					throw new IllegalStateException("buffer already released");
				}
			}
			aquired = false;
			delegate.releaseBufferForSharingNoEvent(((DebugCommandQueue) cq).delegate);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
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
	
	private static class DebugImage extends Image {
		private final Image delegate;
		private final int flags;
		private final boolean shared;
		private boolean aquired;
		
		public DebugImage(Image delegate, int flags, boolean shared) {
			super(null);
			this.delegate = delegate;
			this.flags = flags;
			this.shared = shared;
		}
		
		private void checkAccess() {
			if (shared && !aquired) {
				throw new IllegalStateException("shared buffer is not aquired");
			}
		}

		@Override
		public Image register() {
			delegate.register();
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
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.readImage(((DebugCommandQueue) cq).delegate, bb, longs, longs1, l, l1);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event readImageAsync(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.readImageAsync(((DebugCommandQueue) cq).delegate, bb, longs, longs1, l, l1), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public void writeImage(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.writeImage(((DebugCommandQueue) cq).delegate, bb, longs, longs1, l, l1);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event writeImageAsync(CommandQueue cq, ByteBuffer bb, long[] longs, long[] longs1, long l, long l1) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.writeImageAsync(((DebugCommandQueue) cq).delegate, bb, longs, longs1, l, l1), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public void copyTo(CommandQueue cq, Image image, long[] longs, long[] longs1, long[] longs2) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(image instanceof DebugImage)) throwWrongType(image);
			((DebugImage) image).checkAccess();
			delegate.copyTo(((DebugCommandQueue) cq).delegate, ((DebugImage) image).delegate, longs, longs1, longs2);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event copyToAsync(CommandQueue cq, Image image, long[] longs, long[] longs1, long[] longs2) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(image instanceof DebugImage)) throwWrongType(image);
			((DebugImage) image).checkAccess();
			Event e = new DebugEvent(delegate.copyToAsync(((DebugCommandQueue) cq).delegate, ((DebugImage) image).delegate, longs, longs1, longs2), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public ImageMapping map(CommandQueue cq, long[] longs, long[] longs1, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			ImageMapping m = delegate.map(((DebugCommandQueue) cq).delegate, longs, longs1, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return m;
		}

		@Override
		public ImageMapping mapAsync(CommandQueue cq, long[] longs, long[] longs1, MappingAccess ma) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			ImageMapping m = delegate.mapAsync(((DebugCommandQueue) cq).delegate, longs, longs1, ma);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return m;
		}

		@Override
		public void unmap(CommandQueue cq, ImageMapping im) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.unmap(((DebugCommandQueue) cq).delegate, im);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
		}

		@Override
		public Event fillAsync(CommandQueue cq, long[] longs, long[] longs1, ColorRGBA crgba) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.fillAsync(((DebugCommandQueue) cq).delegate, longs, longs1, crgba), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event fillAsync(CommandQueue cq, long[] longs, long[] longs1, int[] ints) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.fillAsync(((DebugCommandQueue) cq).delegate, longs, longs1, ints), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event copyToBufferAsync(CommandQueue cq, Buffer buffer, long[] longs, long[] longs1, long l) {
			checkAccess();
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			Event e = new DebugEvent(delegate.copyToBufferAsync(((DebugCommandQueue) cq).delegate, ((DebugBuffer) buffer).delegate, longs, longs1, l), flags);
			if ((flags & FLAG_FINISH_AFTER_MEMOP)!=0) cq.finish();
			return e;
		}

		@Override
		public Event acquireImageForSharingAsync(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to acquire an image that is not shared");
			} else {
				if (aquired) {
					throw new IllegalStateException("image already acquired");
				}
			}
			aquired = true;
			Event e = new DebugEvent(delegate.acquireImageForSharingAsync(((DebugCommandQueue) cq).delegate), flags);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
			return e;
		}

		@Override
		public void acquireImageForSharingNoEvent(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to acquire an image that is not shared");
			} else {
				if (aquired) {
					throw new IllegalStateException("image already acquired");
				}
			}
			aquired = true;
			delegate.acquireImageForSharingNoEvent(((DebugCommandQueue) cq).delegate);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
		}

		@Override
		public Event releaseImageForSharingAsync(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to release an image that is not shared");
			} else {
				if (!aquired) {
					throw new IllegalStateException("image already released");
				}
			}
			aquired = false;
			Event e = new DebugEvent(delegate.releaseImageForSharingAsync(((DebugCommandQueue) cq).delegate), flags);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
			return e;
		}

		@Override
		public void releaseImageForSharingNoEvent(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			if (!shared) {
				throw new IllegalStateException("attempt to release an image that is not shared");
			} else {
				if (!aquired) {
					throw new IllegalStateException("image already released");
				}
			}
			aquired = false;
			delegate.releaseImageForSharingNoEvent(((DebugCommandQueue) cq).delegate);
			if ((flags & FLAG_FINISH_AFTER_ACQUIRE)!=0) cq.finish();
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
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
	
	private static class DebugProgram extends Program {
		private final Program delegate;
		private final int flags;

		public DebugProgram(Program delegate, int flags) {
			super(null);
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public Program register() {
			delegate.register();
			return this;
		}

		@Override
		public void build(String string, Device... devices) throws KernelCompilationException {
			delegate.build(string, devices);
		}

		@Override
		public void build() throws KernelCompilationException {
			delegate.build();
		}

		@Override
		public Kernel createKernel(String string) {
			return new DebugKernel(delegate.createKernel(string), flags);
		}

		@Override
		public Kernel[] createAllKernels() {
			Kernel[] kx = delegate.createAllKernels();
			Kernel[] dkx = new Kernel[kx.length];
			for (int i=0; i<kx.length; ++i) {
				dkx[i] = new DebugKernel(kx[i], flags);
			}
			return dkx;
		}

		@Override
		public ByteBuffer getBinary(Device device) {
			return delegate.getBinary(device);
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
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
	
	private static class DebugKernel extends Kernel {
		private final Kernel delegate;
		private final int flags;

		public DebugKernel(Kernel delegate, int flags) {
			super(null);
			this.delegate = delegate;
			this.flags = flags;
		}

		@Override
		public Kernel register() {
			delegate.register();
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
			delegate.setGlobalWorkSize(ws);
		}

		@Override
		public void setGlobalWorkSize(int i) {
			delegate.setGlobalWorkSize(i);
		}

		@Override
		public void setGlobalWorkSize(int i, int i1) {
			delegate.setGlobalWorkSize(i, i1);
		}

		@Override
		public void setGlobalWorkSize(int i, int i1, int i2) {
			delegate.setGlobalWorkSize(i, i1, i2);
		}

		@Override
		public WorkSize getWorkGroupSize() {
			return delegate.getWorkGroupSize();
		}

		@Override
		public void setWorkGroupSize(WorkSize ws) {
			delegate.setWorkGroupSize(ws);
		}

		@Override
		public void setWorkGroupSize(int i) {
			delegate.setWorkGroupSize(i);
		}

		@Override
		public void setWorkGroupSize(int i, int i1) {
			delegate.setWorkGroupSize(i, i1);
		}

		@Override
		public void setWorkGroupSdize(int i, int i1, int i2) {
			delegate.setWorkGroupSdize(i, i1, i2);
		}

		@Override
		public void setWorkGroupSizeToNull() {
			delegate.setWorkGroupSizeToNull();
		}

		@Override
		public long getMaxWorkGroupSize(Device device) {
			return delegate.getMaxWorkGroupSize(device);
		}

		@Override
		public void setArg(int i, LocalMemPerElement lmpe) {
			delegate.setArg(i, lmpe);
		}

		@Override
		public void setArg(int i, LocalMem lm) {
			delegate.setArg(i, lm);
		}

		@Override
		public void setArg(int i, Buffer buffer) {
			if (!(buffer instanceof DebugBuffer)) throwWrongType(buffer);
			((DebugBuffer) buffer).checkAccess();
			delegate.setArg(i, ((DebugBuffer) buffer).delegate);
		}

		@Override
		public void setArg(int i, Image image) {
			if (!(image instanceof DebugImage)) throwWrongType(image);
			((DebugImage) image).checkAccess();
			delegate.setArg(i, ((DebugImage) image).delegate);
		}

		@Override
		public void setArg(int i, byte b) {
			delegate.setArg(i, b);
		}

		@Override
		public void setArg(int i, short s) {
			delegate.setArg(i, s);
		}

		@Override
		public void setArg(int i, int i1) {
			delegate.setArg(i, i1);
		}

		@Override
		public void setArg(int i, long l) {
			delegate.setArg(i, l);
		}

		@Override
		public void setArg(int i, float f) {
			delegate.setArg(i, f);
		}

		@Override
		public void setArg(int i, double d) {
			delegate.setArg(i, d);
		}

		@Override
		public void setArg(int i, Vector2f vctrf) {
			delegate.setArg(i, vctrf);
		}

		@Override
		public void setArg(int i, Vector4f vctrf) {
			delegate.setArg(i, vctrf);
		}

		@Override
		public void setArg(int i, Quaternion qtrn) {
			delegate.setArg(i, qtrn);
		}

		@Override
		public void setArg(int i, Matrix4f mtrxf) {
			delegate.setArg(i, mtrxf);
		}

		@Override
		public void setArg(int i, Matrix3f mtrxf) {
			delegate.setArg(i, mtrxf);
		}

		@Override
		public void setArg(int i, ByteBuffer bb, long l) {
			delegate.setArg(i, bb, l);
		}

		@Override
		public void setArg(int i, Object o) {
			delegate.setArg(i, o);
		}

		@Override
		public Event Run(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.Run(((DebugCommandQueue) cq).delegate), flags);
			if ((flags & FLAG_FINISH_AFTER_KERNEL)!=0) cq.finish();
			return e;
		}

		@Override
		public void RunNoEvent(CommandQueue cq) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.RunNoEvent(((DebugCommandQueue) cq).delegate);
			if ((flags & FLAG_FINISH_AFTER_KERNEL)!=0) cq.finish();
		}

		private Object[] unwrap(Object[] os) {
			Object[] os2 = new Object[os.length];
			for (int i=0; i<os.length; ++i) {
				if (os[i] instanceof Buffer) {
					if (!(os[i] instanceof DebugBuffer)) throwWrongType(os[i]);
					((DebugBuffer) os[i]).checkAccess();
					os2[i] = ((DebugBuffer) os[i]).delegate;
				} else if (os[i] instanceof Image) {
					if (!(os[i] instanceof DebugImage)) throwWrongType(os[i]);
					((DebugImage) os[i]).checkAccess();
					os2[i] = ((DebugImage) os[i]).delegate;
				} else {
					os2[i] = os[i];
				}
			}
			return os2;
		}
		
		@Override
		public Event Run1(CommandQueue cq, WorkSize ws, Object... os) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.Run1(((DebugCommandQueue) cq).delegate, ws, unwrap(os)), flags);
			if ((flags & FLAG_FINISH_AFTER_KERNEL)!=0) cq.finish();
			return e;
		}

		@Override
		public void Run1NoEvent(CommandQueue cq, WorkSize ws, Object... os) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.Run1NoEvent(((DebugCommandQueue) cq).delegate, ws, unwrap(os));
			if ((flags & FLAG_FINISH_AFTER_KERNEL)!=0) cq.finish();
		}

		@Override
		public Event Run2(CommandQueue cq, WorkSize ws, WorkSize ws1, Object... os) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			Event e = new DebugEvent(delegate.Run2(((DebugCommandQueue) cq).delegate, ws, ws1, unwrap(os)), flags);
			if ((flags & FLAG_FINISH_AFTER_KERNEL)!=0) cq.finish();
			return e;
		}

		@Override
		public void Run2NoEvent(CommandQueue cq, WorkSize ws, WorkSize ws1, Object... os) {
			if (!(cq instanceof DebugCommandQueue)) throwWrongType(cq);
			delegate.Run2NoEvent(((DebugCommandQueue) cq).delegate, ws, ws1, unwrap(os));
			if ((flags & FLAG_FINISH_AFTER_KERNEL)!=0) cq.finish();
		}

		@Override
		public String toString() {
			return delegate.toString();
		}

		@Override
		public void release() {
			delegate.release();
		}

		@Override
		public ObjectReleaser getReleaser() {
			return delegate.getReleaser();
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

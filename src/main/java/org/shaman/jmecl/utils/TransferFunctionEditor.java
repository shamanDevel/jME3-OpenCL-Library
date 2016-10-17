/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.utils;

import com.jme3.app.Application;
import com.jme3.export.JmeExporter;
import com.jme3.export.JmeImporter;
import com.jme3.export.Savable;
import com.jme3.input.MouseInput;
import com.jme3.input.controls.ActionListener;
import com.jme3.input.controls.AnalogListener;
import com.jme3.input.controls.MouseAxisTrigger;
import com.jme3.input.controls.MouseButtonTrigger;
import com.jme3.material.Material;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Vector2f;
import com.jme3.renderer.queue.RenderQueue;
import com.jme3.scene.Geometry;
import com.jme3.scene.Mesh;
import com.jme3.scene.Node;
import com.jme3.scene.Spatial;
import com.jme3.scene.VertexBuffer;
import com.jme3.texture.Image;
import com.jme3.texture.Texture;
import com.jme3.texture.Texture2D;
import com.jme3.texture.image.ColorSpace;
import com.jme3.texture.image.ImageRaster;
import com.jme3.util.BufferUtils;
import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.logging.Logger;

/**
 *
 * @author Sebastian
 */
public class TransferFunctionEditor implements Savable, Serializable {

	private static final Logger LOG = Logger.getLogger(TransferFunctionEditor.class.getName());
	private static final int RESOLUTION = 256;
	private static final int[] LINE_WIDTHS = {2,3};
	private static final int[] POINT_SIZES = {4,5,8};
	private static final String CHANNEL_NAMES[] = {"Red", "Green", "Blue", "Alpha"};
	private static final ColorRGBA CHANNEL_COLORS[] = {
		new ColorRGBA(1, 0, 0, 1),
		new ColorRGBA(0, 1, 0, 1),
		new ColorRGBA(0, 0, 1, 1),
		new ColorRGBA(0.5f, 0.5f, 1, 1)
	};
	private static final float Y_SIZE = 0.8f;

	private static final String KEY_PREFIX = "TFE_";
	private static final String MOUSE_MOVE_Xn = KEY_PREFIX + "x-";
	private static final String MOUSE_MOVE_Xp = KEY_PREFIX + "x+";
	private static final String MOUSE_MOVE_Yn = KEY_PREFIX + "y-";
	private static final String MOUSE_MOVE_Yp = KEY_PREFIX + "y+";
	private static final String MOUSE_CLICK_LEFT = KEY_PREFIX + "clickL";
	private static final String MOUSE_CLICK_RIGHT = KEY_PREFIX + "clickR";
	
	private transient boolean enabled;
	private transient Application app;
	private transient Listener listener;
	private transient Node node;
	private transient Texture[] textures;
	
	private Channel[] channels;
	private int[] order;
	private Channel selectedChannel = null;
	private int selectedPoint = -1;
	private Vector2f mouse = new Vector2f();
	private boolean mouseDragging;
	
	public final static class Channel {
		private final ColorRGBA color;
		private final String name;
		private final int textureIndex;
		private final int textureChannel;
		private final ArrayList<Vector2f> points;
		
		private Node node;
		private Mesh[] meshes = new Mesh[2];
		private Geometry[] geoms = new Geometry[2];
		private VertexBuffer buffer;
		private TransferFunctionEditor tfe;

		/**
		 * A channel of the transfer function
		 * @param color the displayed color
		 * @param name displayed name of this channel
		 * @param textureIndex the index of the texture
		 * @param textureChannel the channel inside the texture
		 */
		public Channel(ColorRGBA color, String name, int textureIndex, int textureChannel) {
			this.color = color;
			this.name = name;
			this.textureIndex = textureIndex;
			this.textureChannel = textureChannel;
			this.points = new ArrayList<>();
			reset();
		}
		
		public void reset() {
			points.clear();
			points.add(new Vector2f(0, 0));
			points.add(new Vector2f(1, 1));
		}
		
		private void init(TransferFunctionEditor tfe, Node node) {
			this.tfe = tfe;
			buffer = new VertexBuffer(VertexBuffer.Type.Position);
			buffer.setupData(VertexBuffer.Usage.Dynamic, 4, VertexBuffer.Format.Float, BufferUtils.createFloatBuffer(4 * points.size()));

			this.node = new Node(name);
			node.attachChild(this.node);
			
			Mesh mesh1 = new Mesh();
			mesh1.setMode(Mesh.Mode.LineStrip);
			mesh1.setBuffer(buffer);
			Material mat1 = new Material(tfe.app.getAssetManager(), "Common/MatDefs/Misc/Unshaded.j3md");
			mat1.setColor("Color", color);
			mat1.getAdditionalRenderState().setLineWidth(LINE_WIDTHS[0]);
			Geometry geom1 = new Geometry(name, mesh1);
			geom1.setMaterial(mat1);
			geom1.setLocalScale(1, Y_SIZE, 1);
			this.node.attachChild(geom1);
			meshes[0] = mesh1;
			geoms[0] = geom1;
			
			Mesh mesh2 = new Mesh();
			mesh2.setMode(Mesh.Mode.Points);
			mesh2.setBuffer(buffer);
			Material mat2 = new Material(tfe.app.getAssetManager(), "org/shaman/jmecl/utils/UnshadedPoint.j3md");
			mat2.getAdditionalRenderState().setPointSprite(true);
			mat2.setColor("Color", color);
			Geometry geom2 = new Geometry(name+"_p", mesh2);
			geom2.setMaterial(mat2);
			geom2.setLocalScale(1, Y_SIZE, 1);
			this.node.attachChild(geom2);
			meshes[1] = mesh2;
			geoms[1] = geom2;
		}
		
		private void updateMesh() {
			geoms[0].getMaterial().getAdditionalRenderState().setLineWidth((tfe.selectedChannel == this) ? LINE_WIDTHS[1] : LINE_WIDTHS[0]);
			FloatBuffer data = BufferUtils.createFloatBuffer(4 * points.size());
			for (int i=0; i<points.size(); ++i) {
				Vector2f p = points.get(i);
				data.put(p.x);
				data.put(p.y);
				data.put(0);
				if (tfe.selectedChannel == this) {
					if (tfe.selectedPoint == i) {
						data.put(POINT_SIZES[2]);
					} else {
						data.put(POINT_SIZES[1]);
					}
				} else {
					data.put(POINT_SIZES[0]);
				}
			}
			data.flip();
			buffer.updateData(data);
			meshes[0].updateCounts();
			meshes[1].updateCounts();
		}
		
		private void updateTexture(ImageRaster raster) {
			for (int i=1; i<points.size(); ++i) {
				Vector2f p1 = points.get(i-1);
				Vector2f p2 = points.get(i);
				int x1 = Math.max(0, (int) Math.floor(p1.x * RESOLUTION));
				int x2 = Math.min(RESOLUTION-1, (int) Math.ceil(p2.x * RESOLUTION));
				ColorRGBA c = new ColorRGBA();
				for (int x=x1; x<=x2; ++x) {
					float v = p1.y + (p2.y - p1.y) * ((x-x1)/(float)(x2-x1));
					c = raster.getPixel(x, 0, c);
					switch (textureChannel) {
						case 0: c.r = v; break;
						case 1: c.g = v; break;
						case 2: c.b = v; break;
						case 3: c.a = v; break;
					}
					raster.setPixel(x, 0, c);
				}
			}
		}
	}

	public TransferFunctionEditor() {
		this(true);
	}

	public TransferFunctionEditor(boolean hasAlphaChannel) {
		if (hasAlphaChannel) {
			channels = new Channel[4];
		} else {
			channels = new Channel[3];
		}
		order = new int[channels.length];
		for (int i=0; i<channels.length; ++i) {
			channels[i] = new Channel(CHANNEL_COLORS[i], CHANNEL_NAMES[i], 0, i);
			order[i] = i;
		}
	}
	
	public TransferFunctionEditor(Channel[] channels) {
		this.channels = channels;
		order = new int[channels.length];
		for (int i=0; i<channels.length; ++i) {
			order[i] = i;
		}
	}
	
	public void initialize(Application app) {
		this.app = app;
		setEnabled(true);
		
		//add input mappings
		app.getInputManager().addMapping(MOUSE_MOVE_Xn, new MouseAxisTrigger(MouseInput.AXIS_X, true));
		app.getInputManager().addMapping(MOUSE_MOVE_Xp, new MouseAxisTrigger(MouseInput.AXIS_X, false));
		app.getInputManager().addMapping(MOUSE_MOVE_Yn, new MouseAxisTrigger(MouseInput.AXIS_Y, true));
		app.getInputManager().addMapping(MOUSE_MOVE_Yp, new MouseAxisTrigger(MouseInput.AXIS_Y, false));
		app.getInputManager().addMapping(MOUSE_CLICK_LEFT, new MouseButtonTrigger(MouseInput.BUTTON_LEFT));
		app.getInputManager().addMapping(MOUSE_CLICK_RIGHT, new MouseButtonTrigger(MouseInput.BUTTON_RIGHT));
		listener = new Listener();
		app.getInputManager().addListener(listener, MOUSE_MOVE_Xn, MOUSE_MOVE_Xp, MOUSE_MOVE_Yn, MOUSE_MOVE_Yp, MOUSE_CLICK_LEFT, MOUSE_CLICK_RIGHT);
		
		//create textures
		int maxIndex = 0;
		for (Channel c : channels) {
			maxIndex = Math.max(maxIndex, c.textureIndex);
		}
		textures = new Texture[maxIndex+1];
		for (int i=0; i<=maxIndex; ++i) {
			ByteBuffer data = BufferUtils.createByteBuffer(4 * RESOLUTION);
			Image image = new Image(Image.Format.ARGB8, RESOLUTION, 1, data, ColorSpace.Linear);
			textures[i] = new Texture2D(image);
		}
		
		node = new Node("TransferFunctionEditor");
		
		//border
		Mesh border = new Mesh();
		border.setMode(Mesh.Mode.LineLoop);
		border.setBuffer(VertexBuffer.Type.Position, 2, new float[]{0,0, 1,0, 1,Y_SIZE, 0,Y_SIZE});
		Material boderMat = new Material(app.getAssetManager(), "Common/MatDefs/Misc/Unshaded.j3md");
		boderMat.setColor("Color", ColorRGBA.White);
		boderMat.getAdditionalRenderState().setLineWidth(LINE_WIDTHS[0]);
		Geometry borderGeom = new Geometry("border", border);
		borderGeom.setMaterial(boderMat);
		borderGeom.setLocalTranslation(0, 0, -1);
		node.attachChild(borderGeom);
		
		for (Channel c : channels) {
			c.init(this, node);
		}
		
		node.setQueueBucket(RenderQueue.Bucket.Gui);
		updateMeshes();
		updateTexture();
	}
	
	private void updateMeshes() {
		for (Channel c : channels) {
			c.updateMesh();
		}
		for (int i=0; i<order.length; ++i) {
			channels[order[i]].node.setLocalTranslation(0, 0, order.length-i);
		}
	}
	
	private void updateTexture() {
		for (int i=0; i<textures.length; ++i) {
			ImageRaster raster = ImageRaster.create(textures[i].getImage());
			for (Channel c : channels) {
				if (c.textureIndex == i) {
					c.updateTexture(raster);
				}
			}
		}
	}

	public boolean isEnabled() {
		return enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}
	
	public Spatial getView() {
		if (app == null) {
			throw new IllegalStateException("not initialized yet");
		}
		return node;
	}
	
	public Texture[] getTextures() {
		if (app == null) {
			throw new IllegalStateException("not initialized yet");
		}
		return textures;
	}
	
	public void reset() {
		for (Channel c : channels) {
			c.reset();
		}
		updateMeshes();
		updateTexture();
	}
	
	private void drag() {
		assert (selectedChannel != null);
		assert (selectedPoint >= 0);
		
		Vector2f newPos = new Vector2f(mouse);
		newPos.x /= node.getWorldScale().x;
		newPos.y /= node.getWorldScale().y * Y_SIZE;
		
		newPos.y = Math.max(0, Math.min(1, newPos.y));
		if (selectedPoint == 0) {
			newPos.x = 0;
		} else if (selectedPoint == selectedChannel.points.size()-1) {
			newPos.x = 1;
		} else {
			float xLeft = selectedChannel.points.get(selectedPoint-1).x;
			float xRight = selectedChannel.points.get(selectedPoint+1).x;
			newPos.x = Math.max(xLeft + (1f/RESOLUTION), Math.min(xRight - (1f/RESOLUTION), newPos.x));
		}
		
		selectedChannel.points.get(selectedPoint).set(newPos);
//		System.out.println("set to "+newPos);
		
		updateMeshes();
		updateTexture();
	}
	
	private boolean checkClick(boolean rightClick) {
		Vector2f size = new Vector2f(node.getWorldScale().x, node.getWorldScale().y * Y_SIZE);
		int maxPointDistance = POINT_SIZES[2];
		maxPointDistance *= maxPointDistance;
		int maxLineDistance = LINE_WIDTHS[1];
		maxLineDistance *= maxLineDistance;
		
		int selectedChannelIndex = -1;
		int selectedChannelIndex2 = -1;
		selectedPoint = -1;
		found:
		for (int i=0; i<order.length; ++i) {
			int j = order[i];
			//check for collision with that channel
			for (int k=0; k<channels[j].points.size(); ++k) {
				Vector2f p = channels[j].points.get(k);
				if (mouse.distanceSquared(p.x*size.x, p.y*size.y) <= maxPointDistance) {
					selectedChannelIndex = j;
					selectedChannelIndex2 = i;
					selectedPoint = k;
					break found;
				}
			}
			for (int k=1; k<channels[j].points.size(); ++k) {
				Vector2f p1 = channels[j].points.get(k-1).clone();
				p1.x *= size.x; p1.y *= size.y;
				Vector2f p2 = channels[j].points.get(k).clone();
				p2.x *= size.x; p2.y *= size.y;
				if (minimum_distance(p1, p2, mouse) <= maxLineDistance) {
					selectedChannelIndex = j;
					selectedChannelIndex2 = i;
					break found;
				}
			}
		}
		
		if (selectedChannelIndex == -1) {
			selectedChannel = null;
			updateMeshes();
//			System.out.println("nothing found");
			return false; //nothing found
		}
		
		if (rightClick) {
			if (selectedPoint == -1) {
				//create new point
				Vector2f newPos = new Vector2f(mouse);
				newPos.x /= node.getWorldScale().x;
				newPos.y /= node.getWorldScale().y * Y_SIZE;
				newPos.y = Math.max(0, Math.min(1, newPos.y));
				for (int i=1; i<channels[selectedChannelIndex].points.size(); ++i) {
					Vector2f p1 = channels[selectedChannelIndex].points.get(i-1);
					Vector2f p2 = channels[selectedChannelIndex].points.get(i);
					if (newPos.x > p1.x && newPos.x < p2.x) {
						channels[selectedChannelIndex].points.add(i, newPos);
						selectedPoint = i;
						break;
					}
				}
			} else {
				//delete selected point
				if (selectedPoint>0 && selectedPoint<channels[selectedChannelIndex].points.size()-1) {
					channels[selectedChannelIndex].points.remove(selectedPoint);
					selectedPoint = -1;
				}
			}
		}
		
		//bring selected channel to the top
		for (int i=selectedChannelIndex2; i>0; --i) {
			order[i] = order[i-1];
		}
		order[0] = selectedChannelIndex;
		selectedChannel = channels[selectedChannelIndex];
		
		updateMeshes();
		
//		System.out.println("selected channel="+selectedChannel.name+" (index="+selectedChannelIndex+", selected point="+selectedPoint);
		
		return selectedPoint >= 0;
	}
	//http://stackoverflow.com/a/1501725
	float minimum_distance(Vector2f v, Vector2f w, Vector2f p) {
		// Return minimum distance between line segment vw and point p
		float l2 = v.distanceSquared(w);  // i.e. |w-v|^2 -  avoid a sqrt
		if (l2 == 0.0) {
			return p.distance(v);   // v == w case
		} 
		// Consider the line extending the segment, parameterized as v + t (w - v).
		// We find projection of point p onto the line. 
		// It falls where t = [(p-v) . (w-v)] / |w-v|^2
		// We clamp t from [0,1] to handle points outside the segment vw.
		float t = Math.max(0, Math.min(1, (p.subtract(v)).dot(w.subtract(v)) / l2));
		Vector2f projection = v.add(w.subtract(v).multLocal(t));  // Projection falls on the segment
		return p.distance(projection);
	}
	
	private void release() {

	}
	
	@Override
	public void write(JmeExporter je) throws IOException {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void read(JmeImporter ji) throws IOException {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}
	
	private class Listener implements AnalogListener, ActionListener {

		@Override
		public void onAnalog(String string, float f, float f1) {
			if (!enabled || app==null) {
				return;
			}
			if (!string.startsWith(KEY_PREFIX)) {
				return;
			}
			mouse.set(app.getInputManager().getCursorPosition());
			mouse.x -= node.getWorldTranslation().x;
			mouse.y -= node.getWorldTranslation().y;
			if (mouseDragging) {
				drag();
			}
		}

		@Override
		public void onAction(String string, boolean bln, float f) {
			if (!enabled || app==null) {
				return;
			}
			if (MOUSE_CLICK_LEFT.equals(string)) {
				if (bln) {
					mouseDragging = checkClick(false);
				} else {
					if (mouseDragging) {
						release();
						mouseDragging = false;
					}
				}
			} else if (MOUSE_CLICK_RIGHT.equals(string)) {
				if (bln) {
					mouseDragging = checkClick(true);
				} else {
					if (mouseDragging) {
						release();
						mouseDragging = false;
					}
				}
			}
		}
	}
}
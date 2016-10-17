/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.test;

import com.jme3.app.SimpleApplication;
import com.jme3.font.BitmapText;
import com.jme3.font.Rectangle;
import com.jme3.input.KeyInput;
import com.jme3.input.controls.ActionListener;
import com.jme3.input.controls.KeyTrigger;
import com.jme3.material.Material;
import com.jme3.material.RenderState;
import com.jme3.math.ColorRGBA;
import com.jme3.opencl.CommandQueue;
import com.jme3.opencl.Context;
import com.jme3.renderer.queue.RenderQueue;
import com.jme3.scene.Geometry;
import com.jme3.scene.Mesh;
import com.jme3.scene.Spatial;
import com.jme3.scene.VertexBuffer;
import com.jme3.scene.shape.Quad;
import java.util.ArrayList;
import org.shaman.jmecl.OpenCLSettings;
import org.shaman.jmecl.fluids.DebugTools;
import org.shaman.jmecl.fluids.FlagGrid;
import org.shaman.jmecl.fluids.FluidSolver;
import org.shaman.jmecl.fluids.Grid;
import org.shaman.jmecl.fluids.MACGrid;
import org.shaman.jmecl.fluids.RealGrid;
import org.shaman.jmecl.utils.DebugContextFactory;
import org.shaman.jmecl.utils.LoggingContextFactory;
import org.shaman.jmecl.utils.SharedTexture;
import org.shaman.jmecl.utils.TransferFunctionEditor;

/**
 *
 * @author Sebastian
 */
public abstract class AbstractFluidTest2D extends SimpleApplication {
	protected static final int X_OFFSET = 250;
	private static final String KEY_PREFIX = "AFT2D_";
	private static final String KEY_RUN = KEY_PREFIX+"Run";
	
	protected Context clContext;
	protected CommandQueue clCommandQueue;
	protected OpenCLSettings clSettings;
	
	private boolean debugContext;
	private boolean loggingContext;
	private boolean recording;
	private boolean running;
	
	private boolean initialized;
	private int resolutionX;
	private int resolutionY;
	protected FluidSolver solver;

	private static class GridWithName<T extends Grid> {
		private final T grid;
		private final String name;
		private GridWithName(T grid, String name) {
			this.grid = grid;
			this.name = name;
		}
	}
	private ArrayList<GridWithName<RealGrid>> realGrids;
	private int realGridSelection = -1;
	private float realGridScale = 1;
	private ArrayList<GridWithName<MACGrid>> macGrids;
	private int macGridSelection = -1;
	private float macGridScale = 1;
	private ArrayList<GridWithName<FlagGrid>> flagGrids;
	private int flagGridSelection = -1;
	
	private boolean updateText = true;
	private BitmapText infoText;
	
	private DebugTools debugTools;
	private Geometry boundsGeometry;
	private SharedTexture realTexture;
	private Material realMaterial;
	private Geometry realGeometry;
	private SharedTexture flagTexture;
	private Material flagMaterial;
	private Geometry flagGeometry;
	private TransferFunctionEditor tfe;
	
	public AbstractFluidTest2D() {
	}

	public boolean isDebugContext() {
		return debugContext;
	}

	protected final void setDebugContext(boolean debugContext) {
		if (initialized) {
			throw new IllegalStateException("solver already initialized");
		}
		this.debugContext = debugContext;
	}

	public boolean isLoggingContext() {
		return loggingContext;
	}

	protected final void setLoggingContext(boolean loggingContext) {
		if (initialized) {
			throw new IllegalStateException("solver already initialized");
		}
		this.loggingContext = loggingContext;
	}

	public int getResolutionX() {
		return resolutionX;
	}

	protected final void setResolutionX(int resolutionX) {
		if (initialized) {
			throw new IllegalStateException("solver already initialized");
		}
		this.resolutionX = resolutionX;
	}

	public int getResolutionY() {
		return resolutionY;
	}

	protected final void setResolutionY(int resolutionY) {
		if (initialized) {
			throw new IllegalStateException("solver already initialized");
		}
		this.resolutionY = resolutionY;
	}

	@Override
	public final void simpleInitApp() {
		clContext = context.getOpenCLContext();
		if (debugContext) {
			clContext = DebugContextFactory.createDebugContext(clContext);
		}
		if (loggingContext) {
			clContext = LoggingContextFactory.createLoggingContext(clContext);
		}
		clCommandQueue = clContext.createQueue();
		clSettings = new OpenCLSettings(clContext, clCommandQueue, null, assetManager);
		
		solver = new FluidSolver(clSettings, resolutionX, resolutionY);
		initialized = true;
		realGrids = new ArrayList<>();
		macGrids = new ArrayList<>();
		flagGrids = new ArrayList<>();
		initSolver(solver);
		
		infoText = new BitmapText(guiFont, false);
		infoText.setSize(guiFont.getCharSet().getRenderedSize());
		infoText.setColor(ColorRGBA.White);
		guiNode.attachChild(infoText);
		
		debugTools = new DebugTools(solver);
		int size = Math.min(settings.getHeight() - 10, settings.getWidth() - X_OFFSET - 5);
		int offsetX = settings.getWidth() - size - 5;
		
		tfe = new TransferFunctionEditor(true);
		tfe.initialize(this);
		Spatial tfeNode = tfe.getView();
		guiNode.attachChild(tfeNode);
		tfeNode.setLocalScale(X_OFFSET-10, 150, 1);
		tfeNode.setLocalTranslation(5, 5, 0);
		
		Mesh boundsMesh = new Mesh();
		boundsMesh.setBuffer(VertexBuffer.Type.Position, 3, new float[]{0,0,0, 1,0,0, 1,1,0, 0,1,0});
		boundsMesh.setMode(Mesh.Mode.LineLoop);
		boundsMesh.updateCounts();
		boundsMesh.updateBound();
		boundsGeometry = new Geometry("bounds", boundsMesh);
		Material boundsMat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
		boundsMat.setColor("Color", ColorRGBA.Blue);
		boundsGeometry.setMaterial(boundsMat);
		boundsGeometry.setLocalTranslation(offsetX, (settings.getHeight()-size)/2-1, 0);
		boundsGeometry.setLocalScale(size+2);
		boundsGeometry.setQueueBucket(RenderQueue.Bucket.Gui);
		guiNode.attachChild(boundsGeometry);
		
		realTexture = debugTools.createRealTexture2D(renderManager);
		realGeometry = new Geometry("realGrid", new Quad(1, 1));
		realMaterial = new Material(assetManager, "org/shaman/jmecl/test/ColorRamped.j3md");
		realMaterial.setTexture("ColorMap", realTexture.getJMETexture());
		realMaterial.setTexture("ColorRamp", tfe.getTextures()[0]);
		realMaterial.getAdditionalRenderState().setFaceCullMode(RenderState.FaceCullMode.Off);
		realMaterial.getAdditionalRenderState().setBlendMode(RenderState.BlendMode.Alpha);
		realGeometry.setMaterial(realMaterial);
		realGeometry.setLocalTranslation(offsetX, (settings.getHeight()-size)/2, 0);
		realGeometry.setLocalScale(size);
		realGeometry.setQueueBucket(RenderQueue.Bucket.Gui);
		guiNode.attachChild(realGeometry);
		
		flagTexture = debugTools.createFlagTexture2D(renderManager);
		flagGeometry = new Geometry("flagGrid", new Quad(1, 1));
		flagMaterial = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
		flagMaterial.setTexture("ColorMap", flagTexture.getJMETexture());
		flagMaterial.getAdditionalRenderState().setFaceCullMode(RenderState.FaceCullMode.Off);
		flagMaterial.getAdditionalRenderState().setBlendMode(RenderState.BlendMode.Alpha);
		flagGeometry.setMaterial(flagMaterial);
		flagGeometry.setLocalTranslation(offsetX, (settings.getHeight()-size)/2, 1);
		flagGeometry.setLocalScale(size);
		flagGeometry.setQueueBucket(RenderQueue.Bucket.Gui);
		guiNode.attachChild(flagGeometry);
		
		inputManager.setCursorVisible(true);
		inputManager.addMapping(KEY_RUN, new KeyTrigger(KeyInput.KEY_R));
		inputManager.addListener(new KeyListener(), KEY_RUN);
	}
	
	protected abstract void initSolver(FluidSolver solver);
	
	protected void setSelectedRealGrid(int index) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		if (index<-1 || index>=realGrids.size()) {
			throw new IllegalArgumentException("index must be between -1 (disable) and num real grids - 1: "+index);
		}
		realGridSelection = index;
		updateText = true;
	}
	
	protected int getSelectedRealGrid() {
		return realGridSelection;
	}
	
	protected int addRealGrid(RealGrid grid, String name) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		realGrids.add(new GridWithName<>(grid, name));
		return realGrids.size()-1;
	}
	
	protected void setRealGridScale(float scale) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		if (scale<=0) {
			throw new IllegalArgumentException("scale must be positive: "+scale);
		}
		realGridScale = scale;
		updateText = true;
	}
	
	protected float getRealGridScale() {
		return realGridScale;
	}
	
	protected void setSelectedMACGrid(int index) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		if (index<-1 || index>=macGrids.size()) {
			throw new IllegalArgumentException("index must be between -1 (disable) and num mac grids - 1: "+index);
		}
		macGridSelection = index;
		updateText = true;
	}
	
	protected int getSelectedMACGrid() {
		return macGridSelection;
	}
	
	protected int addMACGrid(MACGrid grid, String name) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		macGrids.add(new GridWithName<>(grid, name));
		return macGrids.size()-1;
	}
	
	protected void setMACGridScale(float scale) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		if (scale<=0) {
			throw new IllegalArgumentException("scale must be positive: "+scale);
		}
		macGridScale = scale;
		updateText = true;
	}
	
	protected float getMACGridScale() {
		return macGridScale;
	}
	
	protected void setSelectedFlagGrid(int index) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		if (index<-1 || index>=flagGrids.size()) {
			throw new IllegalArgumentException("index must be between -1 (disable) and num flag grids - 1: "+index);
		}
		flagGridSelection = index;
		updateText = true;
	}
	
	protected int getSelectedFlagGrid() {
		return flagGridSelection;
	}
	
	protected int addFlagGrid(FlagGrid grid, String name) {
		if (!initialized) {
			throw new IllegalStateException("not initialized yet");
		}
		flagGrids.add(new GridWithName<>(grid, name));
		return flagGrids.size()-1;
	}

	@Override
	public final void simpleUpdate(float tpf) {
		if (running) {
			updateSolver(tpf);
		}
		inputManager.setCursorVisible(true); //a hack
		
		if (updateText) {
			updateText = false;
			StringBuilder str = new StringBuilder();
			str.append("Screenshot: F12\n");
			str.append("Recording: F11  ").append(recording ? "on" : "off").append('\n');
			str.append("Running: R  ").append(running);
			str.append("\n\n");
			str.append("Real Grid  (Num1,2-3):\n");
			str.append("  ").append(realGridSelection==-1 ? "off" : realGrids.get(realGridSelection).name).append('\n');
			str.append("  scale: ").append(realGridScale);
			str.append("\n\n");
			str.append("MAC Grid  (Num4,5-6):\n");
			str.append("  ").append(macGridSelection==-1 ? "off" : macGrids.get(macGridSelection).name).append('\n');
			str.append("  scale: ").append(macGridScale);
			str.append("\n\n");
			str.append("Flag Grid  (Num7,8-9):\n");
			str.append("  ").append(flagGridSelection==-1 ? "off" : flagGrids.get(flagGridSelection).name).append('\n');
			infoText.setText(str);
			infoText.setLocalTranslation(5, settings.getHeight() - 5, 0);
		}
		
		if (realGridSelection == -1) {
			realGeometry.setCullHint(Spatial.CullHint.Always);
		} else {
			realGeometry.setCullHint(Spatial.CullHint.Never);
			debugTools.fillTextureWithDensity2D(realGrids.get(realGridSelection).grid, realTexture);
		}
		
		if (flagGridSelection == -1) {
			flagGeometry.setCullHint(Spatial.CullHint.Always);
		} else {
			flagGeometry.setCullHint(Spatial.CullHint.Never);
			debugTools.fillTextureWithFlags2D(flagGrids.get(flagGridSelection).grid, flagTexture);
		}
	}
	
	protected abstract void updateSolver(float tpf);
	
	private class KeyListener implements ActionListener {

		@Override
		public void onAction(String string, boolean bln, float f) {
			if (KEY_RUN.equals(string) && bln) {
				running = !running;
				updateText = true;
			}
		}
		
	}
}

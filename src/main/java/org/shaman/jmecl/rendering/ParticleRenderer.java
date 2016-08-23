/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.shaman.jmecl.rendering;

import com.jme3.scene.Geometry;
import com.jme3.scene.Mesh;
import com.jme3.scene.VertexBuffer;

/**
 *
 * @author Sebastian Weiss
 */
public class ParticleRenderer extends Geometry {
	
	private boolean worldSpace;
	
	private static Mesh setupMesh() {
		Mesh mesh = new Mesh();
		mesh.setMode(Mesh.Mode.Points);
		return mesh;
	}

	public ParticleRenderer(String name) {
		super(name, setupMesh());
	}

	public ParticleRenderer() {
		this(null);
	}
	
	public void setParticleCount(int count) {
		VertexBuffer vb = mesh.getBuffer(VertexBuffer.Type.Position);
		vb.getData().limit(count * vb.getNumComponents());
		mesh.updateCounts();
	}
	
	public int getParticleCount() {
		return mesh.getVertexCount();
	}
	
	public int getCapacity() {
		VertexBuffer vb = mesh.getBuffer(VertexBuffer.Type.Position);
		return vb.getData().capacity() / vb.getNumComponents();
	}
	
	/**
     * Returns true if particles should spawn in world space.
     *
     * @return true if particles should spawn in world space.
     *
     * @see ParticleEmitter#setInWorldSpace(boolean)
     */
    public boolean isInWorldSpace() {
        return worldSpace;
    }

    /**
     * Set to true if particles should spawn in world space.
     *
     * <p>If set to true and the particle emitter is moved in the scene,
     * then particles that have already spawned won't be effected by this
     * motion. If set to false, the particles will emit in local space
     * and when the emitter is moved, so are all the particles that
     * were emitted previously.
     *
     * @param worldSpace true if particles should spawn in world space.
     */
    public void setInWorldSpace(boolean worldSpace) {
        this.setIgnoreTransform(worldSpace);
        this.worldSpace = worldSpace;
    }
}

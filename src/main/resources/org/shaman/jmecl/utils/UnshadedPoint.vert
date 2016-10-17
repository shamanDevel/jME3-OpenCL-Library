#import "Common/ShaderLib/GLSLCompat.glsllib"
#import "Common/ShaderLib/Skinning.glsllib"
#import "Common/ShaderLib/Instancing.glsllib"

attribute vec4 inPosition;

uniform float m_PointSize;

void main(){
	gl_PointSize = inPosition.w;
    gl_Position = TransformWorldViewProjection(vec4(inPosition.xyz, 1));
}
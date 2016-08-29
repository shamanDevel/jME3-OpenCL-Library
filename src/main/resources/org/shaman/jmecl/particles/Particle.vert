uniform mat4 g_WorldViewProjectionMatrix;

attribute vec4 inPosition;
//attribute vec4 inColor;
attribute vec4 inTexCoord;

varying vec4 color;

#ifdef USE_TEXTURE
varying vec4 texCoord;
#endif

#ifdef USE_COLOR_RAMP
attribute float inTexCoord3;
uniform sampler2D m_ColorRamp;
#endif

#ifdef EXTRA_ALPHA
uniform float m_Alpha;
#endif

uniform mat4 g_WorldViewMatrix;
uniform mat4 g_WorldMatrix;
uniform vec3 g_CameraPosition;
uniform float m_SizeMultiplier;

void main(){
    vec4 pos = vec4(inPosition.xyz, 1.0);
	float inSize = inPosition.w;

    gl_Position = g_WorldViewProjectionMatrix * pos;
    //color = inColor;
	#ifdef USE_COLOR_RAMP
		color = texture2D(m_ColorRamp, vec2(inTexCoord3, 0.0));
	#else
		color = vec4(1.0);
	#endif
	#ifdef EXTRA_ALPHA
		color.a *= m_Alpha;
	#endif

    #ifdef USE_TEXTURE
        //texCoord = inTexCoord;
		texCoord = vec4(0.0, 0.0, 1.0, 1.0);
    #endif

	vec4 worldPos = g_WorldMatrix * pos;
	float d = distance(g_CameraPosition.xyz, worldPos.xyz);
	float size = (inSize * m_SizeMultiplier) / d;
	gl_PointSize = max(1.0, size);

	//vec4 worldViewPos = g_WorldViewMatrix * pos;
	//gl_PointSize = (inSize * SIZE_MULTIPLIER * m_Quadratic)*100.0 / worldViewPos.z;

	color.a *= min(size, 1.0);
}

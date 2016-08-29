#  if !defined(GL_ES) && __VERSION__ < 120
#    error Point sprite is not supported by the video hardware!
#  endif


#ifdef USE_TEXTURE
uniform sampler2D m_Texture;
varying vec4 texCoord;
#endif

varying vec4 color;

void main(){
    if (color.a <= 0.01)
        discard;

    #ifdef USE_TEXTURE
        vec2 uv = mix(texCoord.xy, texCoord.zw, gl_PointCoord.xy);
        gl_FragColor = texture2D(m_Texture, uv) * color;
		//gl_FragColor = vec4(1.0, uv.x, uv.y, 1.0) * color;
    #else
        gl_FragColor = color;
    #endif

    #ifdef PRE_SHADOW
        if (gl_FragColor.r <= 0.1 && 
            gl_FragColor.g <= 0.1 &&
            gl_FragColor.b <= 0.1) {
            discard;
        }
    #endif
}
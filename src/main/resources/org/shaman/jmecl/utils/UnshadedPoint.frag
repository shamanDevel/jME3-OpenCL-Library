#  if !defined(GL_ES) && __VERSION__ < 120
#    error Point sprite is not supported by the video hardware!
#  endif

uniform vec4 m_Color;

void main(){
	vec4 color = m_Color;

    if (color.a <= 0.01)
        discard;

	vec2 uv = gl_PointCoord.xy;
	uv = (uv-vec2(0.5, 0.5))*2;
	if (uv.x*uv.x + uv.y*uv.y > 1)
		discard;

	gl_FragColor = color;

    #ifdef PRE_SHADOW
        if (gl_FragColor.r <= 0.1 && 
            gl_FragColor.g <= 0.1 &&
            gl_FragColor.b <= 0.1) {
            discard;
        }
    #endif
}
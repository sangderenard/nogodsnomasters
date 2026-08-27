#version 300 es
precision highp float;
precision highp int;

uniform sampler2D turing_feed_4;
layout(location = 0) out vec4 turing_output_0;
layout(location = 1) out vec4 turing_output_1;
layout(location = 2) out vec4 turing_output_2;
layout(location = 3) out vec4 turing_output_3;

void main() {
    ivec2 turing_coordinate = ivec2(gl_FragCoord.xy);
    float v_4 = texelFetch(turing_feed_4, turing_coordinate, 0).r;
    float v_8 = cos(v_4);
    float v_9 = 0.5 * v_8;
    float v_10 = 0.5 + v_9;
    float v_14 = v_4 - 2.0943951023931948;
    float v_16 = cos(v_14);
    float v_17 = 0.5 * v_16;
    float v_18 = 0.5 + v_17;
    float v_22 = v_4 - 4.1887902047863896;
    float v_24 = cos(v_22);
    float v_25 = 0.5 * v_24;
    float v_26 = 0.5 + v_25;
    turing_output_0 = vec4(v_4, 0.0, 0.0, 1.0);
    turing_output_1 = vec4(v_10, 0.0, 0.0, 1.0);
    turing_output_2 = vec4(v_18, 0.0, 0.0, 1.0);
    turing_output_3 = vec4(v_26, 0.0, 0.0, 1.0);
}

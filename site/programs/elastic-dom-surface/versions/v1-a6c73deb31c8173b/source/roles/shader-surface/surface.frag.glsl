#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D turing_dom_state;
uniform int turing_dom_count;
uniform vec2 turing_resolution;
uniform vec2 turing_pointer;
uniform float turing_time;
layout(location = 0) out vec4 turing_output_0;

float roundedBox(vec2 point, vec2 halfExtent, float radius) {
    vec2 q = abs(point) - halfExtent + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

void main() {
    vec2 pixel = vec2(gl_FragCoord.x, turing_resolution.y - gl_FragCoord.y);
    vec3 color = vec3(0.018, 0.025, 0.055);
    float nearest = 1.0e9;
    for (int index = 0; index < 256; ++index) {
        if (index >= turing_dom_count) break;
        vec4 geometry = texelFetch(turing_dom_state, ivec2(0, index), 0);
        vec4 material = texelFetch(turing_dom_state, ivec2(1, index), 0);
        vec4 tint = texelFetch(turing_dom_state, ivec2(2, index), 0);
        float depth = material.x;
        float activity = material.y;
        float radius = min(material.z, min(geometry.z, geometry.w));
        vec2 lightDirection = normalize(vec2(-0.7, -1.0));
        for (int slice = 5; slice >= 0; --slice) {
            float layer = float(slice);
            vec2 extrusion = vec2(layer * 1.8, layer * 2.2) * (1.0 + depth * 0.08);
            vec2 local = pixel - geometry.xy - extrusion;
            float distance = roundedBox(local, geometry.zw, radius);
            if (distance < 0.0 && distance < nearest) {
                float edge = smoothstep(5.0, 0.0, abs(distance));
                vec2 normal = normalize(local / max(geometry.zw, vec2(1.0)));
                float light = 0.48 + 0.34 * max(0.0, dot(normal, lightDirection));
                float top = 1.0 - layer / 5.0;
                vec3 base = mix(tint.rgb * 0.28, tint.rgb, top);
                color = base * light + edge * (0.12 + activity * vec3(0.25, 0.65, 1.0));
                nearest = distance;
            }
        }
    }
    float pointerGlow = exp(-length(pixel - turing_pointer) * 0.018);
    color += pointerGlow * vec3(0.08, 0.24, 0.42);
    color += 0.015 * sin(turing_time * 0.7 + gl_FragCoord.y * 0.012);
    turing_output_0 = vec4(pow(max(color, 0.0), vec3(0.82)), 1.0);
}

#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D turing_dom_state;
uniform int turing_dom_count;
uniform vec2 turing_resolution;
uniform float turing_time;
layout(location = 0) out vec4 turing_output_0;

bool rayBox(
    vec3 rayOrigin, vec3 rayDirection, vec3 center, vec3 halfExtent,
    out float hitDistance, out vec3 hitNormal
) {
    vec3 inverseDirection = 1.0 / rayDirection;
    vec3 first = (center - halfExtent - rayOrigin) * inverseDirection;
    vec3 second = (center + halfExtent - rayOrigin) * inverseDirection;
    vec3 nearAxis = min(first, second);
    vec3 farAxis = max(first, second);
    float nearDistance = max(max(nearAxis.x, nearAxis.y), nearAxis.z);
    float farDistance = min(min(farAxis.x, farAxis.y), farAxis.z);
    if (farDistance < max(nearDistance, 0.0)) return false;
    hitDistance = nearDistance > 0.0 ? nearDistance : farDistance;
    vec3 hit = rayOrigin + rayDirection * hitDistance - center;
    vec3 face = abs(abs(hit) - halfExtent);
    hitNormal = face.x < face.y && face.x < face.z
        ? vec3(sign(hit.x), 0.0, 0.0)
        : (face.y < face.z
            ? vec3(0.0, sign(hit.y), 0.0)
            : vec3(0.0, 0.0, sign(hit.z)));
    return true;
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / turing_resolution) * 2.0 - 1.0;
    float aspect = turing_resolution.x / max(turing_resolution.y, 1.0);
    vec3 rayOrigin = vec3(0.0, 0.0, 8.5);
    vec3 rayDirection = normalize(vec3(ndc.x * aspect, ndc.y, -2.15));
    vec3 lightPosition = vec3(
        3.8 * cos(turing_time * 0.19),
        4.6,
        5.5 + 0.7 * sin(turing_time * 0.13)
    );
    vec3 color = vec3(0.008, 0.012, 0.025);
    float nearest = 1.0e20;

    for (int index = 0; index < 256; ++index) {
        if (index >= turing_dom_count) break;
        vec4 geometry = texelFetch(turing_dom_state, ivec2(0, index), 0);
        vec4 material = texelFetch(turing_dom_state, ivec2(1, index), 0);
        vec4 tint = texelFetch(turing_dom_state, ivec2(2, index), 0);
        vec2 normalizedCenter = geometry.xy / turing_resolution * 2.0 - 1.0;
        vec2 normalizedHalf = geometry.zw / turing_resolution * 2.0;
        vec3 center = vec3(
            normalizedCenter.x * aspect * 4.0,
            normalizedCenter.y * 4.0,
            -0.12 * material.x - float(index) * 0.006
        );
        vec3 halfExtent = vec3(
            max(normalizedHalf.x * aspect * 4.0, 0.018),
            max(normalizedHalf.y * 4.0, 0.018),
            0.10 + 0.025 * min(material.y, 1.0)
        );
        float distance;
        vec3 normal;
        if (rayBox(rayOrigin, rayDirection, center, halfExtent, distance, normal)
            && distance < nearest) {
            vec3 point = rayOrigin + rayDirection * distance;
            vec3 lightVector = lightPosition - point;
            float lightDistance2 = dot(lightVector, lightVector);
            vec3 lightDirection = normalize(lightVector);
            vec3 viewDirection = normalize(rayOrigin - point);
            vec3 halfVector = normalize(lightDirection + viewDirection);
            float diffuse = max(dot(normal, lightDirection), 0.0);
            float specular = pow(max(dot(normal, halfVector), 0.0), 48.0);
            float attenuation = 1.0 / (1.0 + 0.035 * lightDistance2);
            vec3 surface = tint.rgb * (0.13 + 1.15 * diffuse * attenuation);
            surface += vec3(1.0, 0.91, 0.75) * specular * attenuation * 0.9;
            surface += tint.rgb * min(material.y, 1.0) * 0.08;
            color = surface;
            nearest = distance;
        }
    }
    turing_output_0 = vec4(pow(max(color, 0.0), vec3(1.0 / 2.2)), 1.0);
}

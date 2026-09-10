#version 330 core
in vec2 uvs;
out vec4 fragColor;
uniform float aspect_ratio=1;
uniform vec2 mouse;

void main() {
    vec2 adjusted_pos = uvs*vec2(1, aspect_ratio);
    vec2 adjusted_mouse_pos = mouse*vec2(1, aspect_ratio);
    vec3 color = 1-abs(vec3(adjusted_pos-adjusted_mouse_pos, 0));
    float brightness = 1-distance(adjusted_pos, adjusted_mouse_pos);
    fragColor = vec4(color, brightness*brightness*brightness);
}
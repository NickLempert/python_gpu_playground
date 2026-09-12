#version 330 core
in vec2 uvs;
out vec4 fragColor;
uniform vec2 resolution=vec2(1000, 1000);
uniform vec2 position=vec2(0, 0);
uniform float iterations=6;
uniform float zoom=1.0;
uniform float base_height=1.5;
uniform float contrast=0.0;
uniform int seed=0;
uniform float ring_width=2.0;
uniform float ring_distance=10;
uniform float time=0;


int random(int seed){
    int out_seed = seed;
    out_seed *= 3266489917;
    out_seed ^= out_seed >> 15;
    out_seed *= 2246822;
    out_seed ^= out_seed >> 13;
    return out_seed;
}

int combine_seeds(int seeds[20], int length) {
    int out_seed = 0;
    for (int i=0; i<length; i++) {
        out_seed += random(seeds[i]);
    }
    return out_seed;
}

float smoothstep(float x) {
    float y = clamp(x, 0.0, 1.0);
    return y * y * (3.0 - 2.0 * y);
}

float random_float(int seed) {
    return (random(seed)%1000)/1000.0;
}

int sign(float val) {
    if (val < 0) {
        return -1;
    }
    return 1;
}

float perlin(vec2 point_pos,
             vec2 vector_cell_size,
             int seed,
             float strength,
             float bias,
             vec2 cam_pos) {
    float x = point_pos.x;
    float y = point_pos.y;
    x -= cam_pos.x;
    y -= cam_pos.y;

    float sum_x = 0.0;

    float sum_y;
    int vector_cell_x;
    int vector_cell_y;
    float d_x;
    float d_y;
    int seeds[20];
    int local_seed;
    float sign_x;
    float sign_y;
    float v_x;
    float v_y;
    float d;
    float dot_product;
    for (int off_x = 0; off_x<2; off_x++) {
        sum_y = 0.0;
        vector_cell_x = int(x / vector_cell_size.x) + off_x*sign(x);
        d_x = x / vector_cell_size.x - vector_cell_x;
        for (int off_y = 0; off_y<2; off_y++) {
            vector_cell_y = int(y / vector_cell_size.y) + off_y*sign(y);
            d_y = y / vector_cell_size.y - vector_cell_y;
            seeds[0] = seed;
            seeds[1] = vector_cell_x;
            seeds[2] = vector_cell_y;
            local_seed = combine_seeds(seeds, 3);
            if (random(local_seed * 997 + 126) % 2 > 0) {
                sign_x = 1;
            }
            else{
                sign_x = -1;
            }
            if (random(local_seed * 557 + 128) % 2 > 0) {
                sign_y = 1;
            }
            else{
                sign_y = -1;
            }
            v_x = 1 * (random(local_seed * 41 + 12) % 999 + 1) * sign_x;
            v_y = 1 * (random(local_seed * 971 + 124) % 999 + 1) * sign_y;
            d = sqrt(v_x * v_x + v_y * v_y);
            if (d != 0) {
                v_x /= d;
                v_y /= d;
            }
            dot_product = v_x * d_x + v_y * d_y;
            sum_y += dot_product * smoothstep(1 - abs(d_y));
        }
        sum_x += sum_y * smoothstep(1 - abs(d_x));
    }
    return sum_x * strength + bias;
}


float terrain(vec2 adjusted_position) {
    vec2 pixel_pos = uvs * resolution;
    float value = 0;
    float layer;
    float bias;
    float vector_grid_resolution;
    vec2 vector_cell_size;
    float strength;
    for (int i=0; i<iterations; i++) {
        layer = i + 1;
        bias = 0;
        if (layer == 1) {
            bias = base_height / 2;
        }
        vector_grid_resolution = pow(2, layer) / zoom;
        vector_cell_size = vec2(1, 1)*(min(resolution.x, resolution.y) / vector_grid_resolution);

        strength = base_height / pow(2, pow(layer, (1 / (contrast + 1))));
        value += perlin(pixel_pos, vector_cell_size, seed, strength, bias, adjusted_position);
    }
    return value;
}


void main() {
    vec2 adjusted_position = vec2(position.x * zoom + resolution.x / 2, position.y * zoom + resolution.y / 2);

    float value = terrain(adjusted_position);
    bool draw_ring = true;

    float norm_value = value/base_height;
    vec3 color;
    float sample_length = 0.1;
    float sample_value = terrain(adjusted_position+vec2(1, -1)*sample_length);
    if (norm_value < 0.43) {
        if (norm_value < 0.4) {
            color = vec3(0.25, 0.25, 0.6);
        }
        else {
            color = vec3(0.3, 0.6, 0.8);
        }
        draw_ring = false;
        float speeds = 5;
        sample_value = 0;
        for (int speed=1; speed<=speeds; speed++) {
            sample_value += perlin(uvs*resolution,
             vec2(10, 10)*zoom/speed,
              0,
               0.005,
              value-0.0025,
              adjusted_position+vec2(time*speed*speed*speed/20, 0)*zoom)/speeds;
        }
        sample_length = zoom;
    }
    else if (norm_value < 0.45) {
        color = vec3(0.8, 0.8, 0.5);
        draw_ring = false;
    }
    else if (norm_value < 0.6) {
        color = vec3(0.3, 0.5, 0.1);
    }
    else if (norm_value < 0.7) {
        color = vec3(0.7, 0.7, 0.7);
    }
    else {
        color = vec3(1, 1, 1);
    }

    float slope = (value - sample_value)/sample_length;
    float brightness = 1;
    brightness += slope*50*value*zoom;

    float ring = mod(value*255, ring_distance);
    if (draw_ring && ring < ring_width) {
        ring -= 0.5 * ring_width;
        ring /= ring_width / 2 * 1.3;
        color *= smoothstep(abs(ring)+0.3);
    }
    color *= brightness;
    fragColor = vec4(color, 1);
}



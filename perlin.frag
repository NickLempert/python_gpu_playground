#version 330 core
in vec2 uvs;
out vec4 fragColor;
uniform vec2 resolution;

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
        vector_cell_x = int(x / vector_cell_size.x + off_x);
        d_x = x / vector_cell_size.x - vector_cell_x;
        for (int off_y = 0; off_y<2; off_y++) {
            vector_cell_y = int(y / vector_cell_size.y + off_y);
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

void main() {
    // int seeds[20];
    // seeds[0] = int(uvs.x*resolution.x);
    // seeds[1] = int(uvs.y*resolution.y);
    // float random = random_float(combine_seeds(seeds, 2));
    // vec3 color = vec3(random, random, random);
    vec2 pixel_pos = uvs * resolution;
    vec3 color = vec3(1, 1, 1) * perlin(pixel_pos, vec2(100, 100), 0, 1, 0, vec2(0, 0));
    fragColor = vec4(color, 1);
}



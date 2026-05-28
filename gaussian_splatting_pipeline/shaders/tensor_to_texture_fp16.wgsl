// tensor_to_texture.wgsl
// Input buffer is array<u32> where each u32 holds two packed fp16 values.
// Layout: [ch0_all_pixels, ch1_all_pixels, ch2_all_pixels] as fp16 pairs.
// Channels are RGB, stride = width * height pixels.

@group(0) @binding(0) var<storage, read>  inputTensor : array<u32>;
@group(0) @binding(1) var                 outputTex   : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform>        dims        : vec4<u32>;

fn unpack_f16(packed: u32, high: bool) -> f32 {
    let bits = select(packed & 0xFFFFu, (packed >> 16u) & 0xFFFFu, high);

    let sign = (bits >> 15u) & 1u;
    let exp  = (bits >> 10u) & 0x1Fu;
    let mant =  bits         & 0x3FFu;

    if exp == 0u {
        return 0.0;
    }
    if exp == 31u {
        return select(1e38, -1e38, sign == 1u);
    }

    let f = f32(1u << exp) * (1.0 + f32(mant) / 1024.0) / 32768.0;
    return select(f, -f, sign == 1u);
}

@compute @workgroup_size(8, 8, 1)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width  = dims.x;
    let height = dims.y;
    if gid.x >= width || gid.y >= height { return; }

    let pixel_idx = gid.y * width + gid.x;
    let stride    = width * height;

    let r_flat = 0u * stride + pixel_idx;
    let g_flat = 1u * stride + pixel_idx;
    let b_flat = 2u * stride + pixel_idx;

    let r = unpack_f16(inputTensor[r_flat / 2u], (r_flat % 2u) == 1u);
    let g = unpack_f16(inputTensor[g_flat / 2u], (g_flat % 2u) == 1u);
    let b = unpack_f16(inputTensor[b_flat / 2u], (b_flat % 2u) == 1u);

    textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0));
}
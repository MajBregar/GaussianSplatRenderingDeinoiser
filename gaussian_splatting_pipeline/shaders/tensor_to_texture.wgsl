@group(0) @binding(0) var<storage, read> inputTensor: array<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    width: u32,
    height: u32,
    padding0: u32,
    padding1: u32,
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let pixelIndex = y * uniforms.width + x;
    let planeSize = uniforms.width * uniforms.height;

    let r = inputTensor[pixelIndex + 0u * planeSize];
    let g = inputTensor[pixelIndex + 1u * planeSize];
    let b = inputTensor[pixelIndex + 2u * planeSize];

    textureStore(
        outputTexture,
        vec2i(i32(x), i32(y)),
        vec4f(clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0)), 1.0)
    );
}
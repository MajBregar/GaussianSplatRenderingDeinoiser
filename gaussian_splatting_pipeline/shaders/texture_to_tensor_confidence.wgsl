@group(0) @binding(0) var colorTexture:      texture_2d<f32>;
@group(0) @binding(1) var depthTexture:      texture_depth_2d;
@group(0) @binding(2) var confidenceTexture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outputTensor: array<f32>;

struct Uniforms {
    width:    u32,
    height:   u32,
    padding0: u32,
    padding1: u32,
}

@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn compute(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let pixel      = vec2i(i32(x), i32(y));
    let color      = textureLoad(colorTexture,      pixel, 0);
    let depth      = textureLoad(depthTexture,      pixel, 0);
    let confidence = textureLoad(confidenceTexture, pixel, 0);

    let pixelIndex = y * uniforms.width + x;
    let planeSize  = uniforms.width * uniforms.height;

    outputTensor[pixelIndex + 0u * planeSize] = color.r;
    outputTensor[pixelIndex + 1u * planeSize] = color.g;
    outputTensor[pixelIndex + 2u * planeSize] = color.b;
    outputTensor[pixelIndex + 3u * planeSize] = depth;
    outputTensor[pixelIndex + 4u * planeSize] = confidence.r;
}
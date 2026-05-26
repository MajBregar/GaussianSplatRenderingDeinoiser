@group(0) @binding(0) var colorTexture  : texture_2d<f32>;
@group(0) @binding(1) var depthTexture : texture_depth_2d;
@group(0) @binding(2) var motionTexture : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outputTensor : array<f32>;
@group(0) @binding(4) var<uniform> dims : vec4<u32>;

@compute @workgroup_size(8, 8, 1)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width  = dims.x;
    let height = dims.y;

    if (gid.x >= width || gid.y >= height) { return; }

    let coord     = vec2<i32>(i32(gid.x), i32(gid.y));
    let pixel_idx = gid.y * width + gid.x;
    let stride    = width * height;

    let color  = textureLoad(colorTexture,  coord, 0);
    let d = textureLoad(depthTexture, coord, 0);
    let motion = textureLoad(motionTexture, coord, 0);

    let mu = motion.r * 2.0 - 1.0;
    let mv = motion.g * 2.0 - 1.0;

    outputTensor[0u * stride + pixel_idx] = color.r;
    outputTensor[1u * stride + pixel_idx] = color.g;
    outputTensor[2u * stride + pixel_idx] = color.b;
    outputTensor[3u * stride + pixel_idx] = d;
    outputTensor[4u * stride + pixel_idx] = mu;
    outputTensor[5u * stride + pixel_idx] = mv;
}
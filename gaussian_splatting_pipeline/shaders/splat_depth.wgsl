@group(0) @binding(0) var<storage, read_write> depths:  array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;

struct DepthUniforms {
    view:    mat4x4<f32>,
    realN:   u32,
    paddedN: u32,
    _pad0:   u32,
    _pad1:   u32,
}
@group(0) @binding(2) var<uniform> u: DepthUniforms;

struct SplatPosition {
    x:    f32,
    y:    f32,
    z:    f32,
    _pad: f32,
}
@group(0) @binding(3) var<storage, read> positions: array<SplatPosition>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.paddedN) { return; }

    if (i >= u.realN) {
        depths[i]  = -3.402823466e38;
        indices[i] = 0xffffffffu;
        return;
    }

    let p = positions[i];
    indices[i] = i;
    let p4 = vec4f(p.x, p.y, p.z, 1.0);
    let view_pos = u.view * p4;
    depths[i] = -view_pos.z;
}

@group(0) @binding(0) var<storage, read_write> depths:  array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;

struct Uniforms {
    N:    u32,
    k:    u32,
    j:    u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i   = gid.x;
    let ixj = i ^ u.j;

    if (i >= u.N || ixj >= u.N || ixj <= i) { return; }

    let ascending = (i & u.k) == 0u;
    let di        = depths[i];
    let dj        = depths[ixj];

    if ((di <= dj) == ascending) {
        depths[i]   = dj;
        depths[ixj] = di;
        let ti      = indices[i];
        indices[i]  = indices[ixj];
        indices[ixj] = ti;
    }
}

@group(0) @binding(0) var<storage, read_write> depths:  array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;

struct Uniforms {
    N:     u32,
    k:     u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> u: Uniforms;

var<workgroup> sharedDepths:  array<f32, 128>;
var<workgroup> sharedIndices: array<u32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let i  = gid.x;
    let li = lid.x;

    sharedDepths[li]  = depths[i];
    sharedIndices[li] = indices[i];
    workgroupBarrier();

    var j = min(u.k / 2u, 64u);
    while (j >= 1u) {
        let l = li ^ j;
        if (l > li) {
            let ascending  = (i & u.k) == 0u;
            let di         = sharedDepths[li];
            let dl         = sharedDepths[l];
            
            if ((di <= dl) == ascending) {
                sharedDepths[li]  = dl;
                sharedDepths[l]   = di;
                let ti            = sharedIndices[li];
                sharedIndices[li] = sharedIndices[l];
                sharedIndices[l]  = ti;
            }
        }
        j /= 2u;
        workgroupBarrier();
    }

    depths[i]  = sharedDepths[li];
    indices[i] = sharedIndices[li];
}

struct GatherUniforms {
    count:  u32,
    stride: u32,
    _pad0:  u32,
    _pad1:  u32,
}
@group(0) @binding(0) var<uniform>             gatherUniforms: GatherUniforms;
@group(0) @binding(1) var<storage, read>       sortedIndices:  array<u32>;
@group(0) @binding(2) var<storage, read>       srcBuffer:      array<u32>;
@group(0) @binding(3) var<storage, read_write> dstBuffer:      array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dstSlot = gid.x;
    if (dstSlot >= gatherUniforms.count) { return; }

    let srcSlot = sortedIndices[dstSlot];
    if (srcSlot >= gatherUniforms.count) { return; }

    let u32PerSplat = gatherUniforms.stride / 4u;
    let srcBase     = srcSlot  * u32PerSplat;
    let dstBase     = dstSlot  * u32PerSplat;

    for (var off = 0u; off < u32PerSplat; off++) {
        dstBuffer[dstBase + off] = srcBuffer[srcBase + off];
    }
}

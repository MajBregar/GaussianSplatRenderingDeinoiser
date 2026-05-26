struct Uniforms {
    previousVP       : mat4x4<f32>,
    inverseCurrentVP : mat4x4<f32>,
};

@group(0) @binding(0) var depthTexture : texture_depth_2d;
@group(0) @binding(1) var nearestSampler : sampler;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0)       uv       : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let pos = positions[vertexIndex];
    var out : VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv       = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depthTexture, nearestSampler, in.uv);

    let ndcCurrent = vec4<f32>(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,
        depth,
        1.0,
    );

    let worldPos  = uniforms.inverseCurrentVP * ndcCurrent;
    let prevClip  = uniforms.previousVP * (worldPos / worldPos.w);
    let prevNDC   = prevClip.xy / prevClip.w;

    let motion = (ndcCurrent.xy - prevNDC) * 0.5 + 0.5;

    return vec4<f32>(motion, 0.0, 1.0);
}
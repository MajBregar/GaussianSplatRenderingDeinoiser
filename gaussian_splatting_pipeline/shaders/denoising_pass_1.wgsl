@group(0) @binding(0) var colorTexture: texture_2d<f32>;
@group(0) @binding(1) var depthTexture: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

const vertices = array(
    vec2f(-1, -1),
    vec2f( 3, -1),
    vec2f(-1,  3),
);

struct VertexInput {
    @builtin(vertex_index) index: u32,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) texcoords: vec2f,
}

struct FragmentInput {
    @location(0) texcoords: vec2f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clipPosition = vec4f(vertices[input.index], 0.0, 1.0);
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;
    return output;
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    let color = textureSample(colorTexture, textureSampler, input.texcoords);

    // Bound for future denoising logic.
    // Not used in this pass-through version.
    let depth = textureSample(depthTexture, textureSampler, input.texcoords).r;
    _ = depth;

    output.color = color;
    return output;
}
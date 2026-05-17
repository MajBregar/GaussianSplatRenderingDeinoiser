@group(0) @binding(0) var currentColorTexture: texture_2d<f32>;
@group(0) @binding(1) var currentDepthTexture: texture_depth_2d;
@group(0) @binding(2) var historyColorTexture: texture_2d<f32>;
@group(0) @binding(3) var historyDepthTexture: texture_2d<f32>;
@group(0) @binding(4) var historyConfidenceTexture: texture_2d<f32>;
@group(0) @binding(5) var textureSampler: sampler;

struct Uniforms {
    previousViewProjectionMatrix: mat4x4f,
    inverseCurrentViewProjectionMatrix: mat4x4f,

    historyWeight: f32,
    depthThreshold: f32,
    firstFrame: f32,
    maxHistoryConfidence: f32,

    varianceClipGamma: f32,
    colorDifferenceScale: f32,
    reprojectionDistanceScale: f32,
    padding: f32,
}

@group(0) @binding(6) var<uniform> uniforms: Uniforms;

const vertices = array(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
);

struct VertexInput {
    @builtin(vertex_index) index: u32,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) texcoords: vec2f,
}

struct FragmentInput {
    @builtin(position) fragCoord: vec4f,
    @location(0) texcoords: vec2f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) historyColor: vec4f,
    @location(2) historyDepth: f32,
    @location(3) historyConfidence: f32,
    @location(4) debug: vec4<f32>,
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

    let dims = vec2i(textureDimensions(currentColorTexture));
    let currentPixel = clamp(vec2i(input.fragCoord.xy), vec2i(0), dims - vec2i(1));

    let currentColor = textureLoad(currentColorTexture, currentPixel, 0);
    let currentDepth = textureLoad(currentDepthTexture, currentPixel, 0);

    output.color = currentColor;
    output.historyColor = output.color;
    output.historyDepth = currentDepth;
    output.historyConfidence = 0.0;
    output.debug = currentColor;

    return output;
}
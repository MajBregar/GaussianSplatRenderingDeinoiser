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
    padding: f32,
}

@group(0) @binding(6) var<uniform> uniforms: Uniforms;

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
    @location(1) historyColor: vec4f,
    @location(2) historyDepth: f32,
    @location(3) historyConfidence: f32,
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

    let currentColor = textureSample(currentColorTexture, textureSampler, input.texcoords);
    let currentDepth = textureSample(currentDepthTexture, textureSampler, input.texcoords);

    let historyUv = input.texcoords;

    let historyColor = textureSample(historyColorTexture, textureSampler, historyUv);
    let historyDepth = textureSample(historyDepthTexture, textureSampler, historyUv).r;
    let historyConfidence = textureSample(historyConfidenceTexture, textureSampler, historyUv).r;

    let depthDifference = abs(currentDepth - historyDepth);
    let depthValid = depthDifference < uniforms.depthThreshold;
    let useHistory = depthValid && uniforms.firstFrame < 0.5;

    var historyWeight = uniforms.historyWeight;

    if (!useHistory) {
        historyWeight = 0.0;
    }

    let outputColor = mix(currentColor, historyColor, historyWeight);

    var newConfidence = 1.0;

    if (useHistory) {
        newConfidence = min(historyConfidence + 1.0, 32.0);
    }

    output.color = outputColor;
    output.historyColor = outputColor;
    output.historyDepth = currentDepth;
    output.historyConfidence = newConfidence;

    return output;
}
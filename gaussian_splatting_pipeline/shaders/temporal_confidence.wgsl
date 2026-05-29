@group(0) @binding(0) var currentColorTexture: texture_2d<f32>;
@group(0) @binding(1) var currentDepthTexture: texture_depth_2d;
@group(0) @binding(2) var historyColorTexture: texture_2d<f32>;
@group(0) @binding(3) var historyDepthTexture: texture_2d<f32>;
@group(0) @binding(4) var historyConfidenceTexture: texture_2d<f32>;
@group(0) @binding(5) var textureSampler: sampler;

struct Uniforms {
    previousViewProjectionMatrix:        mat4x4f,
    inverseCurrentViewProjectionMatrix:  mat4x4f,

    historyWeight:              f32,
    relativeDepthThreshold:             f32,
    firstFrame:                 f32,
    maxHistoryConfidence:       f32,

    reprojectionDistancePixels:  f32,
    colorHistLower:                   f32,
    colorHistUpper:                   f32,
    padding2:                   f32,
}

@group(0) @binding(6) var<uniform> uniforms: Uniforms;

const vertices = array(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
);

struct VertexInput  { @builtin(vertex_index) index: u32 }
struct VertexOutput { @builtin(position) clipPosition: vec4f, @location(0) texcoords: vec2f }
struct FragmentInput { @builtin(position) fragCoord: vec4f, @location(0) texcoords: vec2f }

struct FragmentOutput {
    @location(0) confidence:        vec4f,
    @location(1) historyColor:      vec4f,
    @location(2) historyDepth:      f32,
    @location(3) historyConfidence: f32,
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clipPosition = vec4f(vertices[input.index], 0.0, 1.0);
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;
    return output;
}

struct ReprojectionResult {
    previousUV:            vec2f,
    previousDepth: f32,
    valid:         bool,
}

fn reprojectCurrentPixelUV(currentUv: vec2f, currentDepth: f32) -> ReprojectionResult {
    var result: ReprojectionResult;
    result.previousUV = currentUv;
    result.previousDepth = currentDepth;
    result.valid = false;

    let currentClip  = vec4f(currentUv.x * 2.0 - 1.0, (1.0 - currentUv.y) * 2.0 - 1.0, currentDepth, 1.0);
    let currentWorld4 = uniforms.inverseCurrentViewProjectionMatrix * currentClip;
    let currentWorld  = currentWorld4.xyz / currentWorld4.w;

    let previousClip = uniforms.previousViewProjectionMatrix * vec4f(currentWorld, 1.0);
    if (previousClip.w <= 0.0) { return result; }

    let previousNDC = previousClip.xyz / previousClip.w;
    let previousUV = vec2f(previousNDC.x * 0.5 + 0.5, 1.0 - (previousNDC.y * 0.5 + 0.5));

    result.previousUV = previousUV;
    result.previousDepth = previousNDC.z;
    result.valid =
        previousNDC.x >= -1.0 && previousNDC.x <= 1.0 &&
        previousNDC.y >= -1.0 && previousNDC.y <= 1.0 &&
        previousNDC.z >=  0.0 && previousNDC.z <= 1.0;

    return result;
}

fn colorVariance(currentColor: vec3f, historyColor: vec3f) -> f32 {
    let diff = abs(currentColor - historyColor);
    return dot(diff, vec3f(0.2126, 0.7152, 0.0722));
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;
    let dims = vec2i(textureDimensions(currentColorTexture));
    let dimsf = vec2f(textureDimensions(currentColorTexture));
    let firstFrame = uniforms.firstFrame >= 0.5;


    let currentFragCoords = vec2i(input.fragCoord.xy);
    let currentColor = textureLoad(currentColorTexture, currentFragCoords, 0);
    let currentDepth = textureLoad(currentDepthTexture, currentFragCoords, 0);
    let currentDepthNonBackground = currentDepth < 0.999999;

    var previousFrameUV = input.texcoords;
    var previousFrameDepth = currentDepth;
    var reprojectionValid = false;

    if (currentDepthNonBackground && !firstFrame) {
        let reprojection = reprojectCurrentPixelUV(input.texcoords, currentDepth);
        previousFrameUV = reprojection.previousUV;
        previousFrameDepth = reprojection.previousDepth;
        reprojectionValid = reprojection.valid;
    }
    let previousFrameFragCoords = clamp(vec2i(previousFrameUV * vec2f(dims)), vec2i(0), dims - vec2i(1));

    let historyDepth = textureLoad(historyDepthTexture, previousFrameFragCoords, 0).r;
    let historyConfidence = textureLoad(historyConfidenceTexture, previousFrameFragCoords, 0).r;

    let historyDepthNonBackground = historyDepth < 0.999999;

    let relativeDepthDifference = abs(previousFrameDepth - historyDepth) / max(abs(historyDepth), 1e-6);
    let depthMask = 1.0 - smoothstep(0.0, uniforms.relativeDepthThreshold, relativeDepthDifference);

    let reprojectionDistancePixels = length((previousFrameUV - input.texcoords) * dimsf);
    let distanceMask = clamp(1.0 - reprojectionDistancePixels / uniforms.reprojectionDistancePixels, 0.0, 1.0);

    let historyColor = textureLoad(historyColorTexture, previousFrameFragCoords, 0).rgb;
    let colorDiff = colorVariance(currentColor.rgb, historyColor);
    let colorStability = 1.0 - smoothstep(uniforms.colorHistLower, uniforms.colorHistUpper, colorDiff);

    let historyUsable =
        select(0.0, 1.0, currentDepthNonBackground)   *
        select(0.0, 1.0, !firstFrame)                 *
        select(0.0, 1.0, reprojectionValid)           *
        select(0.0, 1.0, historyDepthNonBackground)   *
        depthMask                                     *
        distanceMask                                  *
        colorStability;

    let observationSupport = select(1.0, 0.0, !currentDepthNonBackground);
    let carriedConfidence = historyConfidence * historyUsable * uniforms.historyWeight;
    let newConfidence = min(carriedConfidence + observationSupport, uniforms.maxHistoryConfidence);
    let historyBlendFactor = carriedConfidence / max(newConfidence, 1e-6);

    let blendedColor = mix(currentColor.rgb, historyColor, historyBlendFactor);
    let blendedDepth = mix(currentDepth, historyDepth, historyBlendFactor);
    output.historyColor = vec4f(blendedColor, 1.0);
    output.historyDepth = blendedDepth;

    output.historyConfidence = newConfidence;
    output.confidence = vec4f(newConfidence / uniforms.maxHistoryConfidence);

    return output;
}
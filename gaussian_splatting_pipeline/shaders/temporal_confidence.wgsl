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
    depthThreshold:             f32,
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
    @location(2) historyDepth:      vec4f,
    @location(3) historyConfidence: vec4f,
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clipPosition = vec4f(vertices[input.index], 0.0, 1.0);
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;
    return output;
}

fn uvToPixel(uv: vec2f, dims: vec2i) -> vec2i {
    return clamp(vec2i(uv * vec2f(dims)), vec2i(0), dims - vec2i(1));
}

fn isGeometryDepth(depth: f32) -> bool {
    return depth < 0.999999;
}

fn clipToUv(clipPosition: vec4f) -> vec2f {
    let ndc = clipPosition.xyz / clipPosition.w;
    return vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

fn uvIsValid(uv: vec2f) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

fn reconstructWorldPosition(uv: vec2f, depth: f32) -> vec3f {
    let clip = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world = uniforms.inverseCurrentViewProjectionMatrix * clip;
    return world.xyz / world.w;
}

fn loadCurrentDepth(pixel: vec2i) -> f32 {
    let dims = vec2i(textureDimensions(currentDepthTexture));
    return textureLoad(currentDepthTexture, clamp(pixel, vec2i(0), dims - vec2i(1)), 0);
}

fn loadHistoryColorFromUv(uv: vec2f) -> vec4f {
    let dims = vec2i(textureDimensions(historyColorTexture));
    return textureLoad(historyColorTexture, uvToPixel(uv, dims), 0);
}

fn loadHistoryDepthFromUv(uv: vec2f) -> f32 {
    let dims = vec2i(textureDimensions(historyDepthTexture));
    return textureLoad(historyDepthTexture, uvToPixel(uv, dims), 0).r;
}

fn loadHistoryConfidenceFromUv(uv: vec2f) -> f32 {
    let dims = vec2i(textureDimensions(historyConfidenceTexture));
    // decode from rgba8unorm: confidence stored in r channel, normalized by maxHistoryConfidence
    let encoded = textureLoad(historyConfidenceTexture, uvToPixel(uv, dims), 0).r;
    return encoded * uniforms.maxHistoryConfidence;
}

fn encodeConfidence(confidence: f32) -> vec4f {
    let normalized = clamp(confidence / max(uniforms.maxHistoryConfidence, 1.0), 0.0, 1.0);
    return vec4f(normalized, normalized, normalized, 1.0);
}

fn encodeDepth(depth: f32) -> vec4f {
    return vec4f(depth, depth, depth, 1.0);
}

struct ReprojectionResult {
    uv:            vec2f,
    previousDepth: f32,
    valid:         bool,
}

fn computeHistoryUv(currentUv: vec2f, currentDepth: f32) -> ReprojectionResult {
    var result: ReprojectionResult;
    result.uv = currentUv;
    result.previousDepth = currentDepth;
    result.valid = false;

    let worldPosition  = reconstructWorldPosition(currentUv, currentDepth);
    let previousClip   = uniforms.previousViewProjectionMatrix * vec4f(worldPosition, 1.0);

    if (previousClip.w <= 0.0) { return result; }

    let pNDC       = previousClip.xyz / previousClip.w;
    let historyUv  = clipToUv(previousClip);

    result.uv            = historyUv;
    result.previousDepth = pNDC.z;
    result.valid         =
        pNDC.x >= -1.0 && pNDC.x <= 1.0 &&
        pNDC.y >= -1.0 && pNDC.y <= 1.0 &&
        pNDC.z >=  0.0 && pNDC.z <= 1.0 &&
        uvIsValid(historyUv);

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

    let currentPixel = clamp(vec2i(input.fragCoord.xy), vec2i(0), dims - vec2i(1));
    let currentColor = textureLoad(currentColorTexture, currentPixel, 0);
    let currentDepth = loadCurrentDepth(currentPixel);

    let currentGeometry = isGeometryDepth(currentDepth);
    let firstFrame      = uniforms.firstFrame >= 0.5;

    var historyUv          = input.texcoords;
    var reprojectedDepth   = currentDepth;
    var reprojectionValid  = false;

    if (currentGeometry && !firstFrame) {
        let reprojection  = computeHistoryUv(input.texcoords, currentDepth);
        historyUv         = reprojection.uv;
        reprojectedDepth  = reprojection.previousDepth;
        reprojectionValid = reprojection.valid;
    }

    let historyDepth      = loadHistoryDepthFromUv(historyUv);
    let historyConfidence = loadHistoryConfidenceFromUv(historyUv);
    let historyGeometry   = isGeometryDepth(historyDepth);
    let depthDifference   = abs(reprojectedDepth - historyDepth);

    let depthMask = 1.0 - smoothstep(
        uniforms.depthThreshold * 0.75,
        uniforms.depthThreshold,
        depthDifference
    );

    let reprojectionDistancePixels = length((historyUv - input.texcoords) * dimsf);
    let distanceMask = clamp(
        1.0 - reprojectionDistancePixels / uniforms.reprojectionDistancePixels,
        0.0, 1.0
    );

    let historyColor   = loadHistoryColorFromUv(historyUv).rgb;
    let colorDiff      = colorVariance(currentColor.rgb, historyColor);
    let colorStability = 1.0 - smoothstep(uniforms.colorHistLower, uniforms.colorHistUpper, colorDiff);

    let historyUsable =
        select(0.0, 1.0, currentGeometry)   *
        select(0.0, 1.0, !firstFrame)       *
        select(0.0, 1.0, reprojectionValid) *
        select(0.0, 1.0, historyGeometry)   *
        depthMask                           *
        distanceMask                        *
        colorStability;

    let carriedConfidence = historyConfidence * historyUsable * uniforms.historyWeight;
    let newConfidence = min(
        carriedConfidence + 1.0,
        max(uniforms.maxHistoryConfidence, 1.0)
    );


    let finalConfidence = select(newConfidence, uniforms.maxHistoryConfidence, !currentGeometry);

    let blendFactor = carriedConfidence / newConfidence;
    let blendedColor = mix(currentColor.rgb, historyColor, blendFactor);
    let blendedDepth = mix(currentDepth, historyDepth, blendFactor);

    output.historyColor      = vec4f(blendedColor, 1.0);
    output.historyDepth      = encodeDepth(blendedDepth);
    output.confidence        = encodeConfidence(finalConfidence);
    output.historyConfidence = encodeConfidence(newConfidence);
    
    return output;
}
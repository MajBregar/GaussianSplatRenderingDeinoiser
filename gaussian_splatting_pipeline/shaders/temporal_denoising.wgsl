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


const USE_REPROJECTION: bool = true;


const DEBUG_CONFIDENCE: bool = false;
const DEBUG_HISTORY_COLOR: bool = false;
const DEBUG_CURRENT_HISTORY_DIFF: bool = false;
const DEBUG_REJECTION_REASON: bool = false;

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
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.clipPosition = vec4f(vertices[input.index], 0.0, 1.0);
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;

    return output;
}


fn uvToClipPosition(uv: vec2f, depth: f32) -> vec4f {
    return vec4f(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );
}

fn reconstructWorldPosition(uv: vec2f, depth: f32) -> vec3f {
    let clipPosition = uvToClipPosition(uv, depth);
    let worldPosition = uniforms.inverseCurrentViewProjectionMatrix * clipPosition;
    return worldPosition.xyz / worldPosition.w;
}

fn projectWorldToPreviousClip(worldPosition: vec3f) -> vec4f {
    return uniforms.previousViewProjectionMatrix * vec4f(worldPosition, 1.0);
}

fn clipToUv(clipPosition: vec4f) -> vec2f {
    let ndc = clipPosition.xyz / clipPosition.w;

    return vec2f(
        ndc.x * 0.5 + 0.5,
        1.0 - (ndc.y * 0.5 + 0.5)
    );
}

fn uvIsValid(uv: vec2f) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 &&
           uv.y >= 0.0 && uv.y <= 1.0;
}

fn uvToPixel(uv: vec2f, dims: vec2i) -> vec2i {
    return clamp(
        vec2i(uv * vec2f(dims)),
        vec2i(0),
        dims - vec2i(1)
    );
}

fn isGeometryDepth(depth: f32) -> bool {
    return depth < 0.999999;
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn loadCurrentColor(pixel: vec2i) -> vec4f {
    let dims = vec2i(textureDimensions(currentColorTexture));
    let p = clamp(pixel, vec2i(0), dims - vec2i(1));
    return textureLoad(currentColorTexture, p, 0);
}

fn loadCurrentDepth(pixel: vec2i) -> f32 {
    let dims = vec2i(textureDimensions(currentDepthTexture));
    let p = clamp(pixel, vec2i(0), dims - vec2i(1));
    return textureLoad(currentDepthTexture, p, 0);
}

fn loadHistoryColorFromUv(uv: vec2f) -> vec4f {
    let dims = vec2i(textureDimensions(historyColorTexture));
    let pixel = uvToPixel(uv, dims);
    return textureLoad(historyColorTexture, pixel, 0);
}

fn loadHistoryDepthFromUv(uv: vec2f) -> f32 {
    let dims = vec2i(textureDimensions(historyDepthTexture));
    let pixel = uvToPixel(uv, dims);
    return textureLoad(historyDepthTexture, pixel, 0).r;
}

fn loadHistoryConfidenceFromUv(uv: vec2f) -> f32 {
    let dims = vec2i(textureDimensions(historyConfidenceTexture));
    let pixel = uvToPixel(uv, dims);
    return textureLoad(historyConfidenceTexture, pixel, 0).r;
}

// ------------------------------------------------------------
// Variance clipping
// ------------------------------------------------------------

fn computeNeighborhoodMean(centerPixel: vec2i) -> vec3f {
    var sum = vec3f(0.0);
    var weightSum = 0.0;

    let centerDepth = loadCurrentDepth(centerPixel);

    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let samplePixel = centerPixel + vec2i(x, y);

            let c = loadCurrentColor(samplePixel).rgb;
            let d = loadCurrentDepth(samplePixel);

            var w = 1.0;

            if (!isGeometryDepth(d)) {
                w = 0.05;
            }

            if (isGeometryDepth(centerDepth) && isGeometryDepth(d)) {
                let depthDelta = abs(centerDepth - d);
                w = w * exp(-depthDelta / max(uniforms.depthThreshold * 4.0, 0.000001));
            }

            sum = sum + c * w;
            weightSum = weightSum + w;
        }
    }

    return sum / max(weightSum, 0.000001);
}

fn computeNeighborhoodVariance(centerPixel: vec2i, mean: vec3f) -> vec3f {
    var sum = vec3f(0.0);
    var weightSum = 0.0;

    let centerDepth = loadCurrentDepth(centerPixel);

    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let samplePixel = centerPixel + vec2i(x, y);

            let c = loadCurrentColor(samplePixel).rgb;
            let d = loadCurrentDepth(samplePixel);

            var w = 1.0;

            if (!isGeometryDepth(d)) {
                w = 0.05;
            }

            if (isGeometryDepth(centerDepth) && isGeometryDepth(d)) {
                let depthDelta = abs(centerDepth - d);
                w = w * exp(-depthDelta / max(uniforms.depthThreshold * 4.0, 0.000001));
            }

            let diff = c - mean;
            sum = sum + diff * diff * w;
            weightSum = weightSum + w;
        }
    }

    return sum / max(weightSum, 0.000001);
}

fn varianceClipHistory(historyColor: vec3f, centerPixel: vec2i) -> vec3f {
    let mean = computeNeighborhoodMean(centerPixel);
    let variance = computeNeighborhoodVariance(centerPixel, mean);
    let sigma = sqrt(max(variance, vec3f(0.0)));

    let minColor = mean - sigma * uniforms.varianceClipGamma;
    let maxColor = mean + sigma * uniforms.varianceClipGamma;

    return clamp(historyColor, minColor, maxColor);
}


struct ReprojectionResult {
    uv: vec2f,
    previousDepth: f32,
    valid: bool,
}

fn computeHistoryUv(currentUv: vec2f, currentDepth: f32) -> ReprojectionResult {
    var result: ReprojectionResult;

    result.uv = currentUv;
    result.previousDepth = currentDepth;
    result.valid = true;

    let worldPosition = reconstructWorldPosition(currentUv, currentDepth);
    let previousClip = projectWorldToPreviousClip(worldPosition);

    if (previousClip.w <= 0.0) {
        result.valid = false;
        return result;
    }

    let pNDC = previousClip.xyz / previousClip.w;
    let historyUv = clipToUv(previousClip);

    let valid =
        pNDC.x >= -1.0 &&
        pNDC.x <=  1.0 &&
        pNDC.y >= -1.0 &&
        pNDC.y <=  1.0 &&
        pNDC.z >=  0.0 &&
        pNDC.z <=  1.0 &&
        uvIsValid(historyUv);

    result.uv = historyUv;
    result.previousDepth = pNDC.z;
    result.valid = valid;
    return result;
}


fn computeBasicHistoryFactor(historyConfidence: f32) -> f32 {
    let maxConfidence = max(uniforms.maxHistoryConfidence, 1.0);
    let oldConfidence = min(historyConfidence, maxConfidence);
    let newConfidence = oldConfidence + 1.0;

    let confidenceWeight = oldConfidence / newConfidence;

    return clamp(confidenceWeight * uniforms.historyWeight, 0.0, 0.98);
}

fn computeAdaptiveHistoryFactor(
    currentColor: vec3f,
    historyColor: vec3f,
    currentDepthForCompare: f32,
    historyDepth: f32,
    reprojectionDistance: f32,
    historyConfidence: f32
) -> f32 {
    let baseWeight = computeBasicHistoryFactor(historyConfidence);

    let depthDifference = abs(currentDepthForCompare - historyDepth);

    var factor = 1.0;

    factor = factor * (1.0 - smoothstep(
        uniforms.depthThreshold * 0.75,
        uniforms.depthThreshold,
        depthDifference
    ));

    factor = factor * clamp(
        1.0 - reprojectionDistance * uniforms.reprojectionDistanceScale,
        0.25,
        1.0
    );

    return clamp(baseWeight * factor, 0.0, 0.98);
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    let dims = vec2i(textureDimensions(currentColorTexture));
    let currentPixel = clamp(vec2i(input.fragCoord.xy), vec2i(0), dims - vec2i(1));

    let currentColor = textureLoad(currentColorTexture, currentPixel, 0);
    let currentDepth = textureLoad(currentDepthTexture, currentPixel, 0);

    var outputColor = currentColor;
    var newConfidence = 0.0;

    //default output
    output.color = outputColor;
    output.historyColor = outputColor;
    output.historyDepth = currentDepth;
    output.historyConfidence = newConfidence;

    // is background
    if (!isGeometryDepth(currentDepth)) {
        if (DEBUG_REJECTION_REASON) {
            output.color = vec4f(1.0, 0.0, 1.0, 1.0);
            output.historyColor = output.color;
        }
        return output;
    }

    // is first frame
    if (uniforms.firstFrame >= 0.5) {
        if (DEBUG_REJECTION_REASON) {
            output.color = vec4f(1.0, 1.0, 0.0, 1.0);
            output.historyColor = output.color;
        }
        return output;
    }

    
    let reprojection = computeHistoryUv(input.texcoords, currentDepth);

    //invalid reprojection
    if (!reprojection.valid) {
        if (DEBUG_REJECTION_REASON) {
            output.color = vec4f(1.0, 0.0, 0.0, 1.0);
            output.historyColor = output.color;
        }
        return output;
    }

    let historyUv = reprojection.uv;
    let historyColorRaw = loadHistoryColorFromUv(historyUv).rgb;
    let historyDepth = loadHistoryDepthFromUv(historyUv);
    let historyConfidence = loadHistoryConfidenceFromUv(historyUv);

    // reprojection was background previous frame
    if (!isGeometryDepth(historyDepth)) {
        if (DEBUG_REJECTION_REASON) {
            output.color = vec4f(0.0, 0.0, 1.0, 1.0);
            output.historyColor = output.color;
        }
        return output;
    }

    
    let depthDifference = abs(reprojection.previousDepth - historyDepth);

    // reprojection and history buffer depth mismatch
    if (depthDifference > uniforms.depthThreshold) {
        if (DEBUG_REJECTION_REASON) {
            output.color = vec4f(0.0, 1.0, 1.0, 1.0);
            output.historyColor = output.color;
        }
        return output;
    }
    
    // history clip
    var historyColor = varianceClipHistory(historyColorRaw, currentPixel);
    
    // adaptive history weight
    let reprojectionDistance = length(historyUv - input.texcoords);

    var historyFactor = computeBasicHistoryFactor(historyConfidence);

    historyFactor = computeAdaptiveHistoryFactor(
        currentColor.rgb,
        historyColor,
        reprojection.previousDepth,
        historyDepth,
        reprojectionDistance,
        historyConfidence
    );
    
    outputColor = vec4f(mix(currentColor.rgb, historyColor, historyFactor), 1.0);
    newConfidence = min(historyConfidence + 1.0, max(uniforms.maxHistoryConfidence, 1.0));

    // DEBUG OUTPUTS
    if (DEBUG_CONFIDENCE) {
        let confidenceVis = newConfidence / max(uniforms.maxHistoryConfidence, 1.0);
        outputColor = vec4f(vec3f(confidenceVis), 1.0);
    }

    if (DEBUG_HISTORY_COLOR) {
        outputColor = vec4f(historyColorRaw, 1.0);
    }

    if (DEBUG_CURRENT_HISTORY_DIFF) {
        let diff = abs(currentColor.rgb - historyColorRaw);
        outputColor = vec4f(diff * 10.0, 1.0);
    }

    output.color = outputColor;
    output.historyColor = outputColor;
    output.historyDepth = currentDepth;
    output.historyConfidence = newConfidence;

    return output;
}
@group(0) @binding(0) var inputColorTexture: texture_2d<f32>;
@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var confidenceTexture: texture_2d<f32>;
@group(0) @binding(3) var textureSampler: sampler;

struct Uniforms {
    stepSize: f32,
    depthSigma: f32,
    colorSigma: f32,
    maxConfidence: f32,

    baseStrength: f32,
    minSpatialStrength: f32,
    fireflyStrength: f32,
    padding: f32,
}

@group(0) @binding(4) var<uniform> uniforms: Uniforms;

const vertices = array(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
);

const kernel = array<f32, 5>(
    1.0 / 16.0,
    4.0 / 16.0,
    6.0 / 16.0,
    4.0 / 16.0,
    1.0 / 16.0,
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
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.clipPosition = vec4f(vertices[input.index], 0.0, 1.0);
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;

    return output;
}

fn isGeometryDepth(depth: f32) -> bool {
    return depth < 0.999999;
}

fn loadColor(pixel: vec2i) -> vec4f {
    let dims = vec2i(textureDimensions(inputColorTexture));
    let p = clamp(pixel, vec2i(0), dims - vec2i(1));
    return textureLoad(inputColorTexture, p, 0);
}

fn loadDepth(pixel: vec2i) -> f32 {
    let dims = vec2i(textureDimensions(depthTexture));
    let p = clamp(pixel, vec2i(0), dims - vec2i(1));
    return textureLoad(depthTexture, p, 0);
}

fn loadConfidence(pixel: vec2i) -> f32 {
    let dims = vec2i(textureDimensions(confidenceTexture));
    let p = clamp(pixel, vec2i(0), dims - vec2i(1));
    return textureLoad(confidenceTexture, p, 0).r;
}

fn foregroundSupportCount(centerPixel: vec2i) -> i32 {
    var count = 0;

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let d = loadDepth(centerPixel + vec2i(x, y));

            if (isGeometryDepth(d)) {
                count = count + 1;
            }
        }
    }

    return count;
}

fn colorDistance2(a: vec3f, b: vec3f) -> f32 {
    let d = a - b;
    return dot(d, d);
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    let dims = vec2i(textureDimensions(inputColorTexture));
    let centerPixel = clamp(
        vec2i(input.fragCoord.xy),
        vec2i(0),
        dims - vec2i(1)
    );

    let centerColor = loadColor(centerPixel);
    let centerDepth = loadDepth(centerPixel);
    let centerConfidence = loadConfidence(centerPixel);

    // do not run denoising on background
    if (!isGeometryDepth(centerDepth)) {
        output.color = centerColor;
        return output;
    }

    let confidenceNorm = clamp(centerConfidence / max(uniforms.maxConfidence, 1.0), 0.0, 1.0);
    var spatialStrength = uniforms.baseStrength * (1.0 - confidenceNorm);
    spatialStrength = max(spatialStrength, uniforms.minSpatialStrength);

    // firefily handling
    let support = foregroundSupportCount(centerPixel);
    if (support <= 2 && confidenceNorm < 0.35) {
        spatialStrength = max(spatialStrength, uniforms.fireflyStrength);
    }

    var colorSum = vec3f(0.0);
    var weightSum = 0.0;

    let step = i32(uniforms.stepSize);

    for (var ky = 0; ky < 5; ky = ky + 1) {
        for (var kx = 0; kx < 5; kx = kx + 1) {
            let ox = (kx - 2) * step;
            let oy = (ky - 2) * step;

            let samplePixel = centerPixel + vec2i(ox, oy);

            let sampleColor = loadColor(samplePixel);
            let sampleDepth = loadDepth(samplePixel);
            let sampleConfidence = loadConfidence(samplePixel);

            let kernelWeight = kernel[kx] * kernel[ky];

            var depthWeight = 0.0;
            if (isGeometryDepth(sampleDepth)) {
                let depthDelta = abs(centerDepth - sampleDepth);

                let effectiveDepthSigma = max(uniforms.depthSigma * sqrt(max(uniforms.stepSize, 1.0)), 0.000001);
                depthWeight = exp(-depthDelta / effectiveDepthSigma);
            }

            let colorDelta2 = colorDistance2(centerColor.rgb, sampleColor.rgb);

            let effectiveColorSigma = max(uniforms.colorSigma, 0.000001);
            let colorWeight = exp(-colorDelta2 / (effectiveColorSigma * effectiveColorSigma));

            let sampleConfidenceNorm = clamp( sampleConfidence / max(uniforms.maxConfidence, 1.0), 0.0, 1.0);

            // dont fully reject unstable neighbours - give them default confidence
            let confidenceWeight = 0.25 + 0.75 * sampleConfidenceNorm;

            let weight = kernelWeight * depthWeight * colorWeight * confidenceWeight;

            colorSum = colorSum + sampleColor.rgb * weight;
            weightSum = weightSum + weight;
        }
    }

    let filteredColor = colorSum / max(weightSum, 0.000001);

    let finalColor = mix(centerColor.rgb, filteredColor, clamp(spatialStrength, 0.0, 1.0));
    output.color = vec4f(finalColor, 1.0);

    return output;
}
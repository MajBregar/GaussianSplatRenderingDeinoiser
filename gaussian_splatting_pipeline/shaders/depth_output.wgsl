@group(0) @binding(0) var depthTexture: texture_depth_2d;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

struct Uniforms {
    alpha: f32,
    gamma: f32,
    invert: f32,
    showBackground: f32,

    depthMin: f32,
    depthMax: f32,
    contrast: f32,
    padding: f32,
}

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

    let dims = vec2i(textureDimensions(depthTexture));
    let pixel = clamp(vec2i(input.fragCoord.xy), vec2i(0), dims - vec2i(1));
    let depth = textureLoad(depthTexture, pixel, 0);

    // output flat non grayscale color as background
    if (uniforms.showBackground > 0.5 && depth >= 0.999999) {
        output.color = vec4f(1.0, 0.0, 1.0, uniforms.alpha);
        return output;
    }

    let denominator = max(uniforms.depthMax - uniforms.depthMin, 0.000001);

    var visualDepth: f32;
    if (uniforms.invert > 0.5) {
        visualDepth = (uniforms.depthMax - depth) / denominator;
    } else {
        visualDepth = (depth - uniforms.depthMin) / denominator;
    }
    visualDepth = clamp(visualDepth, 0.0, 1.0);

    // visulization contrast
    visualDepth = (visualDepth - 0.5) * uniforms.contrast + 0.5;
    visualDepth = clamp(visualDepth, 0.0, 1.0);

    // non linearity
    visualDepth = pow(visualDepth, max(uniforms.gamma, 0.000001));

    output.color = vec4f(vec3f(visualDepth), uniforms.alpha);
    return output;
}
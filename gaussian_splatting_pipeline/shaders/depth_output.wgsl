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

fn encodeDepth24(depth: f32) -> vec3f {
    let d = clamp(depth, 0.0, 1.0);

    let encoded = vec3f(
        floor(d * 255.0),
        floor(fract(d * 255.0) * 255.0),
        floor(fract(d * 65025.0) * 255.0)
    );

    return encoded / 255.0;
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    let dims = vec2i(textureDimensions(depthTexture));
    let pixel = clamp(vec2i(input.fragCoord.xy), vec2i(0), dims - vec2i(1));

    let depth = textureLoad(depthTexture, pixel, 0);

    output.color = vec4f(encodeDepth24(depth), 1.0);
    return output;
}
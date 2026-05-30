@group(0) @binding(0) var<storage, read>       src    : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst    : array<u32>;
@group(0) @binding(2) var<uniform>             params : vec4<u32>;

// PACK 2 F16 INTO 1 F32 FLOAT FOR TIGHT PACKING IN TENSORS
fn f32_to_f16_bits(val: f32) -> u32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 31u) & 1u;
    let exp  = (bits >> 23u) & 0xFFu;
    let mant =  bits         & 0x7FFFFFu;

    if exp == 0xFFu {
        return (sign << 15u) | 0x7C00u | (select(0u, 0x200u, mant != 0u)); //inf or nan
    }

    let new_exp = i32(exp) - 127 + 15;

    if new_exp >= 31 {
        return (sign << 15u) | 0x7C00u;
    }
    if new_exp <= 0 {
        return (sign << 15u);
    }

    return (sign << 15u) | (u32(new_exp) << 10u) | (mant >> 13u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numel  = params.x;
    let idx    = gid.x * 2u;
    if idx >= numel { return; }

    let lo = f32_to_f16_bits(src[idx]);
    let hi = select(0u, f32_to_f16_bits(src[idx + 1u]), idx + 1u < numel);

    dst[gid.x] = lo | (hi << 16u);
}
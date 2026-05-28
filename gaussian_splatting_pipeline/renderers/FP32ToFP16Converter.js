import code from 'shaders/fp32_to_fp16.wgsl?raw';

export class FP32ToFP16Converter {
    constructor(device) {
        this.device = device;

        this.layout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
            compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
        });

        this.uniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    convert(f32Buffer, f16Buffer, numel) {
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Uint32Array([numel, 0, 0, 0]));

        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: f32Buffer } },
                { binding: 1, resource: { buffer: f16Buffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numel / 2 / 256));
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
}
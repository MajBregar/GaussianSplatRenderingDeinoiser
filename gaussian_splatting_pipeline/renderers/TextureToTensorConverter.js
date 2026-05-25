const code = await fetch(
    new URL('shaders/texture_to_tensor.wgsl', import.meta.url)
).then(response => response.text());

export class TextureToTensorConverter {
    constructor(device) {
        this.device = device;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'float' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'depth' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' },
                },
            ],
        });

        const module = this.device.createShaderModule({ code });

        this.pipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.layout],
            }),
            compute: {
                module,
                entryPoint: 'compute',
            },
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    render(colorTexture, depthTexture, outputTensorBuffer, width, height) {
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            new Uint32Array([
                width,
                height,
                0,
                0,
            ])
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                {
                    binding: 0,
                    resource: colorTexture.createView(),
                },
                {
                    binding: 1,
                    resource: depthTexture.createView(),
                },
                {
                    binding: 2,
                    resource: {
                        buffer: outputTensorBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(width / 8),
            Math.ceil(height / 8),
            1
        );

        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
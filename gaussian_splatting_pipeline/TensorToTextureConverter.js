const code = await fetch(
    new URL('./shaders/tensor_to_texture.wgsl', import.meta.url)
).then(response => response.text());

export class TensorToTextureConverter {
    constructor(device, outputFormat = 'rgba8unorm') {
        this.device = device;
        this.outputFormat = outputFormat;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: 'write-only',
                        format: this.outputFormat,
                    },
                },
                {
                    binding: 2,
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

    render(inputTensorBuffer, outputTexture, width, height) {
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
                    resource: {
                        buffer: inputTensorBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: outputTexture.createView(),
                },
                {
                    binding: 2,
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
const code = await fetch(new URL('./shaders/depth_output.wgsl', import.meta.url))
    .then(response => response.text());

export class DepthCompositor {
    constructor(device, format = 'rgba8unorm') {
        this.device = device;
        this.format = format;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'depth' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {},
                },
            ],
        });

        const module = this.device.createShaderModule({ code });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.layout],
            }),
            vertex: {
                module,
            },
            fragment: {
                module,
                targets: [{
                    format: this.format,
                    // No blending: fragment output directly replaces render target.
                }],
            },
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.alpha = 1.0;
        this.gamma = 1.0;
        this.invert = true;
        this.showBackground = true;

        this.depthMin = 0.98;
        this.depthMax = 1.0;
        this.contrast = 1.0;
    }

    render(renderTarget, depthTexture, alpha = this.alpha) {
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            new Float32Array([
                alpha,
                this.gamma,
                this.invert ? 1.0 : 0.0,
                this.showBackground ? 1.0 : 0.0,

                this.depthMin,
                this.depthMax,
                this.contrast,
                0.0,
            ])
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                {
                    binding: 0,
                    resource: depthTexture.createView(),
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: renderTarget.color.createView(),
                loadOp: 'clear',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
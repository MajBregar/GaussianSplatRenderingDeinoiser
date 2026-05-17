const code = await fetch(new URL('./shaders/denoising_pass_1.wgsl', import.meta.url))
    .then(response => response.text());

export class Denoiser {

    constructor(device, format = 'rgba16float') {
        this.device = device;
        this.format = format;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    //color texture
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float' },
                },
                {
                    //depth texture
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float' },
                },
                {
                    //sampler
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'non-filtering' },
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
                }],
            },
        });

        this.sampler = this.device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        });
    }

    render(renderTarget, colorTexture, depthTexture) {
        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                { binding: 0, resource: colorTexture.createView() },
                { binding: 1, resource: depthTexture.createView() },
                { binding: 2, resource: this.sampler },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: renderTarget.color.createView(),
                loadOp: 'clear',
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
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
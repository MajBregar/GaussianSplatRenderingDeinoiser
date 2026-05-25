import code from 'shaders/spatial_denoising.wgsl?raw';

export class SpatialDenoiser {
    constructor(device, format = 'rgba16float') {
        this.device = device;
        this.format = format;

        this.width = 0;
        this.height = 0;

        this.pingTexture = null;
        this.pongTexture = null;

        // Default à-trous pass schedule.
        // Start with 3 passes. Add 8 later if needed.
        this.steps = [1, 2, 4, 8];
        this.passStrengths = [1.0, 0.75, 0.5, 0.25];

        this.depthSigma = 0.02;
        this.colorSigma = 0.35;
        this.maxConfidence = 24.0;
        this.baseStrength = 1.0;
        this.minSpatialStrength = 0.0;
        this.fireflyStrength = 0.85;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    // Input color from previous pass or temporal denoiser.
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' },
                },
                {
                    // Current renderer depth.
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'depth' },
                },
                {
                    // Temporal confidence/history length.
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float' },
                },
                {
                    // Kept for structural similarity with other passes.
                    // Shader uses textureLoad, so this is currently not required.
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'non-filtering' },
                },
                {
                    binding: 4,
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
                }],
            },
        });

        this.sampler = this.device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        });

        // 8 floats = 32 bytes.
        this.uniformBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    resize(width, height) {
        if (this.width === width && this.height === height) return;

        this.width = width;
        this.height = height;

        this.pingTexture?.destroy();
        this.pongTexture?.destroy();

        this.pingTexture = this.#createColorTexture();
        this.pongTexture = this.#createColorTexture();
    }

    render(renderTarget, colorTexture, depthTexture, confidenceTexture) {
        if (!this.pingTexture || !this.pongTexture) {
            throw new Error('SpatialDenoiser.resize(width, height) must be called before render().');
        }

        if (this.steps.length === 0) {
            throw new Error('SpatialDenoiser.steps must contain at least one pass step.');
        }

        const commandEncoder = this.device.createCommandEncoder();

        let inputTexture = colorTexture;

        for (let passIndex = 0; passIndex < this.steps.length; passIndex++) {
            const isLastPass = passIndex === this.steps.length - 1;

            const outputTexture = isLastPass
                ? renderTarget.color
                : (passIndex % 2 === 0 ? this.pingTexture : this.pongTexture);

            const stepSize = this.steps[passIndex];
            const passStrength = this.passStrengths[Math.min(passIndex, this.passStrengths.length - 1)];

            this.device.queue.writeBuffer(
                this.uniformBuffer,
                0,
                new Float32Array([
                    stepSize,
                    this.depthSigma,
                    this.colorSigma,
                    this.maxConfidence,

                    this.baseStrength * passStrength,
                    this.minSpatialStrength,
                    this.fireflyStrength,
                    0.0,
                ])
            );

            const bindGroup = this.device.createBindGroup({
                layout: this.layout,
                entries: [
                    {
                        binding: 0,
                        resource: inputTexture.createView(),
                    },
                    {
                        binding: 1,
                        resource: depthTexture.createView(),
                    },
                    {
                        binding: 2,
                        resource: confidenceTexture.createView(),
                    },
                    {
                        binding: 3,
                        resource: this.sampler,
                    },
                    {
                        binding: 4,
                        resource: {
                            buffer: this.uniformBuffer,
                        },
                    },
                ],
            });

            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: outputTexture.createView(),
                    loadOp: 'clear',
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    storeOp: 'store',
                }],
            });

            renderPass.setPipeline(this.pipeline);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(3);
            renderPass.end();

            inputTexture = outputTexture;
        }

        this.device.queue.submit([commandEncoder.finish()]);
    }

    #createColorTexture() {
        return this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }
}
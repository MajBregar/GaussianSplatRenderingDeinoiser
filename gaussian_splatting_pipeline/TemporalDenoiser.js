import { vec3, mat4 } from 'glm';
import { getGlobalViewMatrix, getProjectionMatrix } from 'engine/core/SceneUtils.js';

const code = await fetch(new URL('./shaders/temporal_denoising.wgsl', import.meta.url)).then(response => response.text());

export class TemporalDenoiser {
    constructor(device, format = 'rgba16float') {
        this.device = device;
        this.format = format;

        this.width = 0;
        this.height = 0;

        this.historyColorTexture = null;
        this.nextHistoryColorTexture = null;

        this.historyDepthTexture = null;
        this.nextHistoryDepthTexture = null;

        this.historyConfidenceTexture = null;
        this.nextHistoryConfidenceTexture = null;

        this.previousViewProjectionMatrix = null;
        this.firstFrame = true;

        this.historyWeight = 1.0;
        this.maxHistoryConfidence = 24.0;
        this.depthThreshold = 0.02;

        this.varianceClipGamma = 5.0;
        this.colorDifferenceScale = 0.0; // unused 
        this.reprojectionDistanceScale = 2.0;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
            ],
        });

        const module = this.device.createShaderModule({ code });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
            vertex: { module },
            fragment: {
                module,
                targets: [
                    { format: this.format },
                    { format: this.format },
                    { format: 'r32float' },
                    { format: 'r32float' },
                ],
            },
        });

        this.sampler = this.device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 176,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    resize(width, height) {
        if (this.width === width && this.height === height) return;

        this.width = width;
        this.height = height;

        this.historyColorTexture?.destroy();
        this.nextHistoryColorTexture?.destroy();
        this.historyDepthTexture?.destroy();
        this.nextHistoryDepthTexture?.destroy();
        this.historyConfidenceTexture?.destroy();
        this.nextHistoryConfidenceTexture?.destroy();

        this.historyColorTexture = this.#createColorHistoryTexture();
        this.nextHistoryColorTexture = this.#createColorHistoryTexture();

        this.historyDepthTexture = this.#createScalarHistoryTexture();
        this.nextHistoryDepthTexture = this.#createScalarHistoryTexture();

        this.historyConfidenceTexture = this.#createScalarHistoryTexture();
        this.nextHistoryConfidenceTexture = this.#createScalarHistoryTexture();

        this.reset();
    }

    reset() {
        this.firstFrame = true;
        this.previousViewProjectionMatrix = null;
    }

    render(renderTarget, colorTexture, depthTexture, camera) {
        if (!this.historyColorTexture) {
            throw new Error('TemporalDenoiser.resize(width, height) must be called before render().');
        }

        const viewMatrix = getGlobalViewMatrix(camera);
        const projectionMatrix = getProjectionMatrix(camera);

        const currentViewProjectionMatrix = mat4.create();
        mat4.multiply(currentViewProjectionMatrix, projectionMatrix, viewMatrix);

        const inverseCurrentViewProjectionMatrix = mat4.create();
        mat4.invert(inverseCurrentViewProjectionMatrix, currentViewProjectionMatrix);

        if (!this.previousViewProjectionMatrix) {
            this.previousViewProjectionMatrix = new Float32Array(currentViewProjectionMatrix);
        }

        const uniformData = new Float32Array(44);

        uniformData.set(this.previousViewProjectionMatrix, 0);
        uniformData.set(inverseCurrentViewProjectionMatrix, 16);

        uniformData[32] = this.historyWeight;
        uniformData[33] = this.depthThreshold;
        uniformData[34] = this.firstFrame ? 1.0 : 0.0;
        uniformData[35] = this.maxHistoryConfidence;

        uniformData[36] = this.varianceClipGamma;
        uniformData[37] = this.colorDifferenceScale;
        uniformData[38] = this.reprojectionDistanceScale;
        uniformData[39] = 0.0;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                { binding: 0, resource: colorTexture.createView() },
                { binding: 1, resource: depthTexture.createView() },
                { binding: 2, resource: this.historyColorTexture.createView() },
                { binding: 3, resource: this.historyDepthTexture.createView() },
                { binding: 4, resource: this.historyConfidenceTexture.createView() },
                { binding: 5, resource: this.sampler },
                { binding: 6, resource: { buffer: this.uniformBuffer } },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: renderTarget.color.createView(),
                    loadOp: 'clear',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    storeOp: 'store',
                },
                {
                    view: this.nextHistoryColorTexture.createView(),
                    loadOp: 'clear',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    storeOp: 'store',
                },
                {
                    view: this.nextHistoryDepthTexture.createView(),
                    loadOp: 'clear',
                    clearValue: { r: 1, g: 0, b: 0, a: 0 },
                    storeOp: 'store',
                },
                {
                    view: this.nextHistoryConfidenceTexture.createView(),
                    loadOp: 'clear',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    storeOp: 'store',
                },
            ],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        this.#swapHistoryTextures();

        this.previousViewProjectionMatrix = new Float32Array(currentViewProjectionMatrix);
        this.firstFrame = false;
    }


    #createColorHistoryTexture() {
        return this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #createScalarHistoryTexture() {
        return this.device.createTexture({
            size: [this.width, this.height],
            format: 'r32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #swapHistoryTextures() {
        [this.historyColorTexture, this.nextHistoryColorTexture] = [this.nextHistoryColorTexture, this.historyColorTexture];
        [this.historyDepthTexture, this.nextHistoryDepthTexture] = [this.nextHistoryDepthTexture, this.historyDepthTexture];
        [this.historyConfidenceTexture, this.nextHistoryConfidenceTexture] = [this.nextHistoryConfidenceTexture, this.historyConfidenceTexture];
    }
}
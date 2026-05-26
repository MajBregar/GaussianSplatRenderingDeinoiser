import { mat4 } from 'glm';
import { getGlobalViewMatrix, getProjectionMatrix } from 'engine/core/SceneUtils.js';

import motion_map_code from 'shaders/motion_map.wgsl?raw';

export class MotionMapCompositor {
    constructor(device, format = 'rgba8unorm') {
        this.device = device;
        this.format = format;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
            ],
        });

        const module = this.device.createShaderModule({ code: motion_map_code });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
            vertex: { module },
            fragment: {
                module,
                targets: [{ format: this.format }],
            },
        });

        this.sampler = this.device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        });

        // 2x mat4 = 32 floats = 128 bytes
        this.uniformBuffer = this.device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    render(renderTarget, depthTexture, currentVP, previousVP) {
        const inverseCurrentVP = mat4.create();
        mat4.invert(inverseCurrentVP, currentVP);

        const uniformData = new Float32Array(32);
        uniformData.set(previousVP,       0);
        uniformData.set(inverseCurrentVP, 16);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                { binding: 0, resource: depthTexture.createView() },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: { buffer: this.uniformBuffer } },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view:       renderTarget.color.createView(),
                loadOp:     'clear',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                storeOp:    'store',
            }],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
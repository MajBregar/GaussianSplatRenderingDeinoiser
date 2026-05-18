import { mat4 } from 'glm';
import { Camera } from 'engine/core.js';

import {
    getLocalModelMatrix,
    getGlobalViewMatrix,
    getProjectionMatrix,
} from 'engine/core/SceneUtils.js';

import { createVertexBuffer } from 'engine/core/VertexUtils.js';
import { Splat } from './file_handling/Splat.js';
import { SplatSorter } from './SplatSorter.js';

export class SplatRenderer {
    constructor(device, code, format = 'rgba8unorm', sorted = false) {
        this.device = device;
        this.format = format;
        this.sorted = sorted;
        this.gpuObjects = new WeakMap();

        if (sorted) {
            this.sorter = new SplatSorter(device);
        }

        const module = this.device.createShaderModule({ code });

        this.instanceBufferLayout = {
            arrayStride: 48,
            stepMode: 'instance',
            attributes: [
                { name: 'position',  shaderLocation: 1, offset: 0,  format: 'float32x3' },
                { name: 'color',     shaderLocation: 2, offset: 12, format: 'unorm8x4' },
                { name: 'rotation',  shaderLocation: 3, offset: 16, format: 'float32x4' },
                { name: 'scale',     shaderLocation: 4, offset: 32, format: 'float32x3' },
            ],
        };

        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module,
                buffers: [this.instanceBufferLayout],
            },
            fragment: {
                module,
                targets: [
                    this.sorted
                        ? {
                            format: this.format,
                            blend: {
                                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            },
                        }
                        : { format: this.format },
                ],
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: this.sorted ? 'always' : 'less',
                format: 'depth24plus',
            },
            primitive: { topology: 'triangle-strip' },
        });

        this.splatScale = 3;
        this.loBound   = 0;
        this.hiBound   = 1;
        this.gamma     = 1;
    }

    prepareSplat(splat) {
        if (this.gpuObjects.has(splat)) return this.gpuObjects.get(splat);

        const instanceBufferArrayBuffer = createVertexBuffer(splat.splats, this.instanceBufferLayout);

        const instanceBuffer = this.device.createBuffer({
            size:  instanceBufferArrayBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(instanceBuffer, 0, instanceBufferArrayBuffer);

        let sortedInstanceBuffer = null;
        if (this.sorted) {
            this.sorter.prepare(splat, instanceBufferArrayBuffer);
            sortedInstanceBuffer = this.sorter.getSortedBuffer(splat);
        }

        const splatUniformBuffer = this.device.createBuffer({
            size:  96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const splatBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: splatUniformBuffer } }],
        });

        const gpuObjects = { instanceBuffer, sortedInstanceBuffer, splatUniformBuffer, splatBindGroup };
        this.gpuObjects.set(splat, gpuObjects);

        return gpuObjects;
    }

    prepareCamera(camera) {
        if (this.gpuObjects.has(camera)) return this.gpuObjects.get(camera);

        const cameraUniformBuffer = this.device.createBuffer({
            size:  144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const cameraBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: cameraUniformBuffer } }],
        });

        const gpuObjects = { cameraUniformBuffer, cameraBindGroup };
        this.gpuObjects.set(camera, gpuObjects);
        return gpuObjects;
    }

    render(renderTarget, scene, camera) {
        const viewMatrix = getGlobalViewMatrix(camera);

        if (this.sorted) {
            const sortEncoder = this.device.createCommandEncoder();
            this.#recordSortPasses(sortEncoder, scene, mat4.create(), viewMatrix);
            this.device.queue.submit([sortEncoder.finish()]);
        }

        const commandEncoder = this.device.createCommandEncoder();

        this.renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view:       renderTarget.color.createView(),
                loadOp:     'clear',
                clearValue: [1, 1, 1, 1],
                storeOp:    'store',
            }],
            depthStencilAttachment: {
                view:            renderTarget.depth.createView(),
                depthLoadOp:     'clear',
                depthClearValue: 1,
                depthStoreOp:    'store',
            },
        });

        this.renderPass.setPipeline(this.pipeline);

        const projectionMatrix = getProjectionMatrix(camera);
        const cameraComponent  = camera.getComponentOfType(Camera);
        const { cameraUniformBuffer, cameraBindGroup } = this.prepareCamera(cameraComponent);

        this.device.queue.writeBuffer(cameraUniformBuffer, 0,   viewMatrix);
        this.device.queue.writeBuffer(cameraUniformBuffer, 64,  projectionMatrix);
        this.device.queue.writeBuffer(cameraUniformBuffer, 128, new Float32Array([
            1 / renderTarget.color.width  ** 2,
            1 / renderTarget.color.height ** 2,
        ]));
        this.renderPass.setBindGroup(0, cameraBindGroup);

        this.renderNode(scene, mat4.create(), camera);

        this.renderPass.end();
        this.device.queue.submit([commandEncoder.finish()]);
        this.renderPass = null;
    }

    #recordSortPasses(encoder, node, modelMatrix, viewMatrix) {
        const globalModelMatrix = mat4.mul(mat4.create(), modelMatrix, getLocalModelMatrix(node));

        for (const splat of node.getComponentsOfType(Splat)) {
            this.prepareSplat(splat);
            const modelViewMatrix = mat4.mul(mat4.create(), viewMatrix, globalModelMatrix);
            this.sorter.recordSort(encoder, splat, modelViewMatrix);
        }

        for (const child of node.children) {
            this.#recordSortPasses(encoder, child, globalModelMatrix, viewMatrix);
        }
    }

    renderNode(node, modelMatrix = mat4.create(), camera) {
        const globalModelMatrix = mat4.mul(mat4.create(), modelMatrix, getLocalModelMatrix(node));

        for (const splat of node.getComponentsOfType(Splat)) {
            this.renderSplat(splat, globalModelMatrix);
        }

        for (const child of node.children) {
            this.renderNode(child, globalModelMatrix, camera);
        }
    }

    renderSplat(splat, modelMatrix) {
        const { instanceBuffer, sortedInstanceBuffer, splatUniformBuffer, splatBindGroup }
            = this.prepareSplat(splat);

        this.device.queue.writeBuffer(splatUniformBuffer, 0, modelMatrix);
        this.device.queue.writeBuffer(splatUniformBuffer, 64, new Float32Array([
            this.splatScale, this.loBound, this.hiBound, performance.now(), this.gamma,
        ]));

        this.renderPass.setBindGroup(1, splatBindGroup);
        this.renderPass.setVertexBuffer(0, this.sorted ? sortedInstanceBuffer : instanceBuffer);
        this.renderPass.draw(4, splat.splats.length);
    }
}
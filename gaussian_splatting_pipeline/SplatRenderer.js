import { mat4 } from 'glm';
import { Camera } from 'engine/core.js';

import {
    getLocalModelMatrix,
    getGlobalViewMatrix,
    getProjectionMatrix,
} from 'engine/core/SceneUtils.js';

import { createVertexBuffer } from 'engine/core/VertexUtils.js';
import { Splat } from './file_handling/Splat.js';

export class SplatRenderer {
    constructor(device, code, format = 'rgba8unorm', sorted = false) {
        this.device = device;
        this.format = format;
        this.sorted = sorted;
        this.gpuObjects = new WeakMap();

        const module = this.device.createShaderModule({ code });

        this.instanceBufferLayout = {
            arrayStride: 48,
            stepMode: 'instance',
            attributes: [
                { name: 'position', shaderLocation: 1, offset: 0, format: 'float32x3' },
                { name: 'color', shaderLocation: 2, offset: 12, format: 'unorm8x4' },
                { name: 'rotation', shaderLocation: 3, offset: 16, format: 'float32x4' },
                { name: 'scale', shaderLocation: 4, offset: 32, format: 'float32x3' },
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
                                color: {
                                    srcFactor: 'one',
                                    dstFactor: 'one-minus-src-alpha',
                                    operation: 'add',
                                },
                                alpha: {
                                    srcFactor: 'one',
                                    dstFactor: 'one-minus-src-alpha',
                                    operation: 'add',
                                },
                            },
                        }
                        : {
                            format: this.format,
                        },
                ],
            },
            depthStencil: {
                depthWriteEnabled: !this.sorted,
                depthCompare: this.sorted ? 'always' : 'less',
                format: 'depth24plus',
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });

        this.splatScale = 3;
        this.loBound = 0;
        this.hiBound = 1;
        this.gamma = 1;
    }

    prepareSplat(splat) {
        if (this.gpuObjects.has(splat)) {
            return this.gpuObjects.get(splat);
        }

        const instanceBufferArrayBuffer = createVertexBuffer(
            splat.splats,
            this.instanceBufferLayout
        );

        const instanceBuffer = this.device.createBuffer({
            size: instanceBufferArrayBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(instanceBuffer, 0, instanceBufferArrayBuffer);

        let sortedInstanceBuffer = null;
        let sortedInstanceBufferArrayBuffer = null;
        let sortedInstanceBufferBytes = null;
        let instanceBufferBytes = null;
        let sortEntries = null;

        if (this.sorted) {
            sortedInstanceBufferArrayBuffer = new ArrayBuffer(instanceBufferArrayBuffer.byteLength);
            sortedInstanceBufferBytes = new Uint8Array(sortedInstanceBufferArrayBuffer);
            instanceBufferBytes = new Uint8Array(instanceBufferArrayBuffer);

            sortedInstanceBuffer = this.device.createBuffer({
                size: instanceBufferArrayBuffer.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });

            sortEntries = new Array(splat.splats.length);

            for (let i = 0; i < splat.splats.length; i++) {
                sortEntries[i] = {
                    index: i,
                    z: 0,
                    position: splat.splats[i].position,
                };
            }

            sortedInstanceBufferBytes.set(instanceBufferBytes);
            this.device.queue.writeBuffer(
                sortedInstanceBuffer,
                0,
                sortedInstanceBufferArrayBuffer
            );
        }

        const splatUniformBuffer = this.device.createBuffer({
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const splatBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: splatUniformBuffer },
                },
            ],
        });

        const gpuObjects = {
            instanceBuffer,
            instanceBufferArrayBuffer,

            sortedInstanceBuffer,
            sortedInstanceBufferArrayBuffer,
            sortedInstanceBufferBytes,
            instanceBufferBytes,
            sortEntries,

            splatUniformBuffer,
            splatBindGroup,
        };

        this.gpuObjects.set(splat, gpuObjects);
        return gpuObjects;
    }

    prepareCamera(camera) {
        if (this.gpuObjects.has(camera)) {
            return this.gpuObjects.get(camera);
        }

        const cameraUniformBuffer = this.device.createBuffer({
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const cameraBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: cameraUniformBuffer },
                },
            ],
        });

        const gpuObjects = {
            cameraUniformBuffer,
            cameraBindGroup,
        };

        this.gpuObjects.set(camera, gpuObjects);
        return gpuObjects;
    }

    render(renderTarget, scene, camera) {
        const commandEncoder = this.device.createCommandEncoder();

        this.renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: renderTarget.color.createView(),
                    loadOp: 'clear',
                    clearValue: [0, 0, 0, 1],
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: renderTarget.depth.createView(),
                depthLoadOp: 'clear',
                depthClearValue: 1,
                depthStoreOp: 'store',
            },
        });

        this.renderPass.setPipeline(this.pipeline);

        const viewMatrix = getGlobalViewMatrix(camera);
        const projectionMatrix = getProjectionMatrix(camera);
        const cameraComponent = camera.getComponentOfType(Camera);

        const { cameraUniformBuffer, cameraBindGroup } = this.prepareCamera(cameraComponent);

        this.device.queue.writeBuffer(cameraUniformBuffer, 0, viewMatrix);
        this.device.queue.writeBuffer(cameraUniformBuffer, 64, projectionMatrix);

        const minSplatSizeInPixels = 1;
        const screenResolutionInvSq = new Float32Array([
            minSplatSizeInPixels / renderTarget.color.width ** 2,
            minSplatSizeInPixels / renderTarget.color.height ** 2,
        ]);

        this.device.queue.writeBuffer(cameraUniformBuffer, 128, screenResolutionInvSq);
        this.renderPass.setBindGroup(0, cameraBindGroup);

        this.renderNode(scene, mat4.create(), camera);

        this.renderPass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        this.renderPass = null;
    }

    renderNode(node, modelMatrix = mat4.create(), camera) {
        const localMatrix = getLocalModelMatrix(node);
        const globalModelMatrix = mat4.mul(mat4.create(), modelMatrix, localMatrix);

        const splats = node.getComponentsOfType(Splat);

        for (const splat of splats) {
            this.renderSplat(splat, globalModelMatrix, camera);
        }

        for (const child of node.children) {
            this.renderNode(child, globalModelMatrix, camera);
        }
    }

    renderSplat(splat, modelMatrix, camera) {
        const gpuObjects = this.prepareSplat(splat);

        const {
            instanceBuffer,
            splatUniformBuffer,
            splatBindGroup,
        } = gpuObjects;

        this.device.queue.writeBuffer(splatUniformBuffer, 0, modelMatrix);

        this.device.queue.writeBuffer(
            splatUniformBuffer,
            64,
            new Float32Array([
                this.splatScale,
                this.loBound,
                this.hiBound,
                performance.now(),
                this.gamma,
            ])
        );

        const activeInstanceBuffer = this.sorted
            ? this.#updateAndGetSortedInstanceBuffer(gpuObjects, modelMatrix, camera)
            : instanceBuffer;

        this.renderPass.setBindGroup(1, splatBindGroup);
        this.renderPass.setVertexBuffer(0, activeInstanceBuffer);
        this.renderPass.draw(4, splat.splats.length);
    }

    #updateAndGetSortedInstanceBuffer(gpuObjects, modelMatrix, camera) {
        const viewMatrix = getGlobalViewMatrix(camera);

        const modelViewMatrix = mat4.create();
        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);

        this.#updateSortDepths(gpuObjects.sortEntries, modelViewMatrix);

        gpuObjects.sortEntries.sort((a, b) => {
            return a.z - b.z;
        });

        this.#reorderInstanceBufferBytes(gpuObjects);

        this.device.queue.writeBuffer(
            gpuObjects.sortedInstanceBuffer,
            0,
            gpuObjects.sortedInstanceBufferArrayBuffer
        );

        return gpuObjects.sortedInstanceBuffer;
    }

    #updateSortDepths(sortEntries, modelViewMatrix) {
        for (let i = 0; i < sortEntries.length; i++) {
            const entry = sortEntries[i];
            const p = entry.position;

            const x = p[0];
            const y = p[1];
            const z = p[2];

            entry.z =
                modelViewMatrix[2] * x +
                modelViewMatrix[6] * y +
                modelViewMatrix[10] * z +
                modelViewMatrix[14];
        }
    }

    #reorderInstanceBufferBytes(gpuObjects) {
        const stride = this.instanceBufferLayout.arrayStride;

        const source = gpuObjects.instanceBufferBytes;
        const destination = gpuObjects.sortedInstanceBufferBytes;
        const sortEntries = gpuObjects.sortEntries;

        for (let dstIndex = 0; dstIndex < sortEntries.length; dstIndex++) {
            const srcIndex = sortEntries[dstIndex].index;

            const srcOffset = srcIndex * stride;
            const dstOffset = dstIndex * stride;

            for (let byteOffset = 0; byteOffset < stride; byteOffset++) {
                destination[dstOffset + byteOffset] = source[srcOffset + byteOffset];
            }
        }
    }
}
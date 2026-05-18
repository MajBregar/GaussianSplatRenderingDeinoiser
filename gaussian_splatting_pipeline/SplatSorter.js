const depthShaderCode = await fetch(
    new URL('./shaders/splat_depth.wgsl', import.meta.url)
).then(r => r.text());

const bitonicGlobalShaderCode = await fetch(
    new URL('./shaders/splat_bitonic_global.wgsl', import.meta.url)
).then(r => r.text());

const bitonicLocalShaderCode = await fetch(
    new URL('./shaders/splat_bitonic_local.wgsl', import.meta.url)
).then(r => r.text());

const gatherShaderCode = await fetch(
    new URL('./shaders/splat_gather.wgsl', import.meta.url)
).then(r => r.text());

function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

export class SplatSorter {
    constructor(device) {
        this.device = device;
        this._gpuObjects = new WeakMap();

        this.depthPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: depthShaderCode }), entryPoint: 'main' },
        });
        this.bitonicGlobalPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: bitonicGlobalShaderCode }), entryPoint: 'main' },
        });
        this.bitonicLocalPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: bitonicLocalShaderCode }), entryPoint: 'main' },
        });
        this.gatherPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: gatherShaderCode }), entryPoint: 'main' },
        });
    }

    prepare(splat, instanceBufferArrayBuffer) {
        if (this._gpuObjects.has(splat)) return this._gpuObjects.get(splat);

        const realN   = splat.splats.length;
        const paddedN = nextPow2(realN);
        const device  = this.device;

        const sortPassDefs = this.computeSortPasses(paddedN);

        const positionData = new Float32Array(paddedN * 4);
        for (let i = 0; i < realN; i++) {
            const p = splat.splats[i].position;
            positionData[i * 4 + 0] = p[0];
            positionData[i * 4 + 1] = p[1];
            positionData[i * 4 + 2] = p[2];
            positionData[i * 4 + 3] = 0;
        }



        const positionBuffer = device.createBuffer({
            size:  positionData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(positionBuffer, 0, positionData);

        const depthBuffer = device.createBuffer({
            size:  paddedN * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const indexBuffer = device.createBuffer({
            size:  paddedN * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        const srcInstanceBuffer = device.createBuffer({
            size:  instanceBufferArrayBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(srcInstanceBuffer, 0, instanceBufferArrayBuffer);

        const dstInstanceBuffer = device.createBuffer({
            size:  instanceBufferArrayBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
        });

        const depthUniformBuffer = device.createBuffer({
            size:  80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(depthUniformBuffer, 64, new Uint32Array([realN, paddedN, 0, 0]));

        const gatherUniformBuffer = device.createBuffer({
            size:  16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gatherUniformBuffer, 0, new Uint32Array([realN, 48, 0, 0]));

        const depthBindGroup = device.createBindGroup({
            layout: this.depthPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: depthBuffer } },
                { binding: 1, resource: { buffer: indexBuffer } },
                { binding: 2, resource: { buffer: depthUniformBuffer } },
                { binding: 3, resource: { buffer: positionBuffer } },
            ],
        });

        const sortPasses = sortPassDefs.map(({ type, k, j }) => {
            const uniformBuffer = device.createBuffer({
                size:  16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([paddedN, k, j, 0]));

            const pipeline = type === 'global' ? this.bitonicGlobalPipeline : this.bitonicLocalPipeline;
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: depthBuffer } },
                    { binding: 1, resource: { buffer: indexBuffer } },
                    { binding: 2, resource: { buffer: uniformBuffer } },
                ],
            });

            return { type, pipeline, bindGroup };
        });

        const gatherBindGroup = device.createBindGroup({
            layout: this.gatherPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: gatherUniformBuffer } },
                { binding: 1, resource: { buffer: indexBuffer } },
                { binding: 2, resource: { buffer: srcInstanceBuffer } },
                { binding: 3, resource: { buffer: dstInstanceBuffer } },
            ],
        });

        const gpuObjects = {
            realN, paddedN,
            depthUniformBuffer,
            depthBindGroup, sortPasses, gatherBindGroup,
            dstInstanceBuffer,
        };

        this._gpuObjects.set(splat, gpuObjects);
        return gpuObjects;
    }

    computeSortPasses(paddedN) {
        const passes = [];
        let k = 2;
        while (k <= paddedN) {
            let j = k >> 1;
            while (j >= 128) {
                passes.push({ type: 'global', k, j });
                j >>= 1;
            }
            passes.push({ type: 'local', k, j: 0 });
            k <<= 1;
        }
        return passes;
    }

    recordSort(encoder, splat, modelViewMatrix) {
        const g = this._gpuObjects.get(splat);
        if (!g) throw new Error('SplatSorter: splat not prepared');

        const { realN, paddedN, depthUniformBuffer,
                depthBindGroup, sortPasses, gatherBindGroup } = g;
        
        const workgroup_size = 128;
        const groups = Math.ceil(paddedN / workgroup_size);

        this.device.queue.writeBuffer(depthUniformBuffer, 0, modelViewMatrix);

        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.depthPipeline);
            pass.setBindGroup(0, depthBindGroup);
            pass.dispatchWorkgroups(groups);
            pass.end();
        }

        for (const { pipeline, bindGroup } of sortPasses) {
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(groups);
            pass.end();
        }

        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.gatherPipeline);
            pass.setBindGroup(0, gatherBindGroup);
            pass.dispatchWorkgroups(Math.ceil(realN / workgroup_size));
            pass.end();
        }
    }

    getSortedBuffer(splat) {
        return this._gpuObjects.get(splat)?.dstInstanceBuffer ?? null;
    }
}
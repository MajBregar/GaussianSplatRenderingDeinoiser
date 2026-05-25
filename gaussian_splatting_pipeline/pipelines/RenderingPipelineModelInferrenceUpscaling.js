import { Camera } from 'engine/core.js';
import { SplatRenderer } from 'renderers/SplatRenderer.js';
import { Compositor } from 'renderers/Compositor.js';

import { TextureToTensorConverter } from 'renderers/TextureToTensorConverter.js';
import { TensorToTextureConverter } from 'renderers/TensorToTextureConverter.js';
import * as ort from 'onnxruntime-web/webgpu';

import { PerformanceTracker } from '../PerformanceTracker.js';
import { OnnxModelInitializer } from '../OnnxModelInitializer.js';

import stochastic_splatting_code from 'shaders/stochastic_splat_render.wgsl?raw';

export class RenderingPipelineModelInferrenceUpscaling {
    constructor({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        onnxModel,
        performanceTracker
    }) {
        this.device  = device;
        this.context = context;
        this.format  = format;
        this.perf = performanceTracker ?? new PerformanceTracker();

        this.INPUT_WIDTH  = 640;
        this.INPUT_HEIGHT = 360;

        this.splatFormat  = 'rgba8unorm';
        this.baseChannels = 32;
        this.pad_factor   = 32;

        this.canvas = canvas;
        this.scene  = scene;
        this.camera = camera;

        this.renderer                 = new SplatRenderer(device, stochastic_splatting_code, this.splatFormat, false);
        this.compositor               = new Compositor(device, format);
        this.textureToTensorConverter = new TextureToTensorConverter(device);
        this.tensorToTextureConverter = new TensorToTextureConverter(device);

        //360p
        this.noisyColorTexture      = null;
        this.noisyDepthTexture      = null;
        this.inferenceInputBuffer   = null;

        //720p
        this.inferenceOutputTexture = null;

        this.inferenceInFlight       = false;
        this.last_render_timestamp   = 0;
        this.last_render_duration_ms = 0;
        this.completed_render_count  = 0;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 'r' && !event.repeat) {
                console.log('[Pipeline] reset hidden state');
                this.#resetHidden();
            }
        });

        this.onnx_model        = onnxModel;
        this.hiddenStates      = {};
        this.hiddenReady       = false;
        this.hiddenInputNames  = [];
        this.hiddenOutputNames = [];
        this.runOptions        = {};

        this.hiddenInputNames  = this.onnx_model.session.inputNames.filter(n => /^h\d+_in$/.test(n)).sort();
        this.hiddenOutputNames = this.hiddenInputNames.map(n => n.replace('_in', '_out'));
        this.onnxModelReady    = true;

        console.log('[Pipeline] inputMetadata:',  this.onnx_model.session.handler.inputMetadata);
        console.log('[Pipeline] outputMetadata:', this.onnx_model.session.handler.outputMetadata);
    }

    update(_t, _dt) {}

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
        this.#resetHidden();
    }

    async render() {
        if (!this.onnxModelReady)   return false;
        if (this.inferenceInFlight) return false;

        this.inferenceInFlight = true;
        const t0 = performance.now();

        try {
            await this.#renderFrame();
            this.last_render_timestamp   = performance.now();
            this.last_render_duration_ms = performance.now() - t0;
            this.completed_render_count++;
            return true;
        } catch (err) {
            console.error('[Pipeline] frame FAILED:', err);
            return false;
        } finally {
            this.inferenceInFlight = false;
        }
    }

    async #renderFrame() {
        this.#ensureResources();

        const inputW = this.INPUT_WIDTH;
        const inputH = this.INPUT_HEIGHT;

        // render noisy at 360p
        this.perf.begin('noisy_render');
        this.renderer.render(
            { color: this.noisyColorTexture, depth: this.noisyDepthTexture },
            this.scene, this.camera
        );
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('noisy_render');

        // convert 360p noisy textures to tensor
        this.perf.begin('texture_to_tensor_noisy');
        this.textureToTensorConverter.render(
            this.noisyColorTexture,
            this.noisyDepthTexture,
            this.inferenceInputBuffer,
            inputW, inputH
        );
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('texture_to_tensor_noisy');

        // build feeds — input is 360p
        this.perf.begin('inferrence_tensor_prep');
        const inputTensor = new ort.Tensor({
            location : 'gpu-buffer',
            gpuBuffer: this.inferenceInputBuffer,
            type     : 'float32',
            dims     : [1, 4, inputH, inputW],
        });

        const feeds = { input: inputTensor };

        for (const name of this.hiddenInputNames) {
            const key = name.replace('_in', '');

            if (this.hiddenReady && this.hiddenStates[key]) {
                const { buffer, dims } = this.hiddenStates[key];
                feeds[name] = new ort.Tensor({
                    location : 'gpu-buffer',
                    gpuBuffer: buffer,
                    type     : 'float32',
                    dims,
                });
            } else {
                const dims = this.#inferHiddenDims(name, inputW, inputH);
                feeds[name] = new ort.Tensor(
                    'float32',
                    new Float32Array(dims.reduce((a, b) => a * b, 1)),
                    dims
                );
            }
        }
        this.perf.end('inferrence_tensor_prep');

        this.perf.begin('inferrence');
        let results;
        try {
            results = await this.onnx_model.session.run(feeds, this.runOptions);
        } finally {
            inputTensor.dispose?.();
            for (const name of this.hiddenInputNames) feeds[name]?.dispose?.();
        }
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('inferrence');

        this.perf.begin('hidden_tensor_output_copy');
        const enc = this.device.createCommandEncoder();
        for (const outName of this.hiddenOutputNames) {
            const key    = outName.replace('_out', '');
            const tensor = results[outName];

            if (!tensor?.gpuBuffer)
                throw new Error(`Hidden state output '${outName}' is not GPU-backed`);

            if (!this.hiddenStates[key]) {
                this.hiddenStates[key] = {
                    buffer: this.#mkBuf(tensor.gpuBuffer.size),
                    dims  : Array.from(tensor.dims),
                };
            }

            enc.copyBufferToBuffer(
                tensor.gpuBuffer, 0,
                this.hiddenStates[key].buffer, 0,
                Math.min(tensor.gpuBuffer.size, this.hiddenStates[key].buffer.size)
            );
            tensor.dispose?.();
        }
        this.device.queue.submit([enc.finish()]);
        this.hiddenReady = true;
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('hidden_tensor_output_copy');

        this.perf.begin('output_tensor_to_texture');
        const outMain = results['output'];
        if (!outMain?.gpuBuffer)
            throw new Error("Main output 'output' is not GPU-backed");

        this.tensorToTextureConverter.render(
            outMain.gpuBuffer,
            this.inferenceOutputTexture,
            this.canvas.width, this.canvas.height
        );
        await this.device.queue.onSubmittedWorkDone();
        outMain.dispose?.();
        this.perf.end('output_tensor_to_texture');

        this.perf.begin('screen_render');
        this.compositor.render(
            { color: this.context.getCurrentTexture() },
            this.inferenceOutputTexture,
            1.0
        );
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('screen_render');
    }

    #inferHiddenDims(inputName, width, height) {
        const pad_factor = this.pad_factor;
        const pH = Math.ceil(height / pad_factor) * pad_factor;
        const pW = Math.ceil(width  / pad_factor) * pad_factor;

        const match = inputName.match(/^h(\d+)_in$/);
        if (!match) throw new Error(`Cannot infer dims for '${inputName}'`);

        const level        = parseInt(match[1], 10) - 1;
        const spatialScale = 1 << level;

        const allMeta = this.onnx_model.session.handler?.inputMetadata;
        const meta    = Array.isArray(allMeta)
            ? allMeta.find(m => m.name === inputName)
            : null;

        const channels = (meta?.shape?.[1] && typeof meta.shape[1] === 'number')
            ? meta.shape[1]
            : this.baseChannels * spatialScale;

        return [1, channels, pH / spatialScale, pW / spatialScale];
    }

    #mkBuf(size) {
        return this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    #resetHidden() {
        for (const { buffer } of Object.values(this.hiddenStates))
            buffer?.destroy();
        this.hiddenStates = {};
        this.hiddenReady  = false;
    }

    #ensureResources() {
        this.#resizeNoisyColorTexture();
        this.#resizeNoisyDepthTexture();
        this.#resizeInputBuffer();
        this.#resizeOutputTexture();
    }

    #resizeNoisyColorTexture() {
        const w = this.INPUT_WIDTH;
        const h = this.INPUT_HEIGHT;
        if (this.noisyColorTexture?.width === w &&
            this.noisyColorTexture?.height === h) return;

        this.noisyColorTexture?.destroy();
        this.noisyColorTexture = this.device.createTexture({
            size  : [w, h],
            format: this.splatFormat,
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }

    #resizeNoisyDepthTexture() {
        const w = this.INPUT_WIDTH;
        const h = this.INPUT_HEIGHT;
        if (this.noisyDepthTexture?.width === w &&
            this.noisyDepthTexture?.height === h) return;

        this.noisyDepthTexture?.destroy();
        this.noisyDepthTexture = this.device.createTexture({
            size  : [w, h],
            format: 'depth24plus',
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }

    #resizeInputBuffer() {
        const size = 1 * 4 * this.INPUT_WIDTH * this.INPUT_HEIGHT * 4;
        if (this.inferenceInputBuffer?.size === size) return;

        this.inferenceInputBuffer?.destroy();
        this.inferenceInputBuffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.#resetHidden();
    }

    #resizeOutputTexture() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.inferenceOutputTexture?.width  === width &&
            this.inferenceOutputTexture?.height === height) return;

        this.inferenceOutputTexture?.destroy();
        this.inferenceOutputTexture = this.device.createTexture({
            size  : [width, height],
            format: this.splatFormat,
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.STORAGE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }
}
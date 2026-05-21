import { Camera } from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';

import { PerformanceTracker } from './PerformanceTracker.js';

import { OnnxModelInitializer } from './OnnxModelInitializer.js';
import { TextureToTensorConverter } from './TextureToTensorConverter.js';
import { TensorToTextureConverter } from './TensorToTextureConverter.js';
import * as ort from 'onnxruntime-web/webgpu';

const stochastic_splatting_code = await fetch(
    new URL('./shaders/stochastic_splat_render.wgsl', import.meta.url)
).then(response => response.text());

export class RenderingPipelineModelInferrence {
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

        this.splatFormat  = 'rgba8unorm';
        this.baseChannels = 32;

        this.canvas = canvas;
        this.scene  = scene;
        this.camera = camera;

        this.renderer                 = new SplatRenderer(device, stochastic_splatting_code, this.splatFormat, false);
        this.compositor               = new Compositor(device, format);
        this.textureToTensorConverter = new TextureToTensorConverter(device);
        this.tensorToTextureConverter = new TensorToTextureConverter(device);

        this.directColorTexture_A   = null;
        this.directDepthTexture_A   = null;
        this.inferenceInputBuffer   = null;
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

        this.onnx_model     = onnxModel;

        this.hiddenStates      = {};
        this.hiddenReady       = false;
        this.hiddenInputNames  = [];
        this.hiddenOutputNames = [];

        this.runOptions = {};
        this.hiddenInputNames  = this.onnx_model.session.inputNames.filter(n => /^h\d+_in$/.test(n)).sort();
        this.hiddenOutputNames = this.hiddenInputNames.map(n => n.replace('_in', '_out'));
        this.onnxModelReady = true;

        // console.log('[Pipeline] hidden inputs :', this.hiddenInputNames);
        // console.log('[Pipeline] hidden outputs:', this.hiddenOutputNames);
        // console.log('[Pipeline] all inputs    :', this.onnx_model.session.inputNames);
        // console.log('[Pipeline] all outputs   :', this.onnx_model.session.outputNames);

        console.log('[Pipeline] inputMetadata:', this.onnx_model.session.handler.inputMetadata);
        console.log('[Pipeline] outputMetadata:', this.onnx_model.session.handler.outputMetadata);
    }

    update(_t, _dt) {}

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
        this.#resetHidden();
    }

    async render() {
        if (!this.onnxModelReady)    return false;
        if (this.inferenceInFlight)  return false;

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
        const width  = this.canvas.width;
        const height = this.canvas.height;

        this.perf.begin('noisy_render');
        this.renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('noisy_render');


        this.perf.begin('texture_to_tensor_noisy');
        this.textureToTensorConverter.render(
            this.directColorTexture_A,
            this.directDepthTexture_A,
            this.inferenceInputBuffer,
            width, height
        );
        await this.device.queue.onSubmittedWorkDone();
        this.perf.end('texture_to_tensor_noisy');


        this.perf.begin('inferrence_tensor_prep');
        const inputTensor = new ort.Tensor({
            location : 'gpu-buffer',
            gpuBuffer: this.inferenceInputBuffer,
            type     : 'float32',
            dims     : [1, 4, height, width],
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
                const dims = this.#inferHiddenDims(name, width, height);
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
            width, height
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
        //only for my current model that pads with multiples of 32
        const pH = Math.ceil(height / 32) * 32;
        const pW = Math.ceil(width  / 32) * 32;

        const match = inputName.match(/^h(\d+)_in$/);
        if (!match) throw new Error(`Cannot infer dims for '${inputName}'`);

        const level = parseInt(match[1], 10) - 1;
        const spatialScale = 1 << level;

        const allMeta = this.onnx_model.session.handler?.inputMetadata;
        const meta = Array.isArray(allMeta)
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
        this.#resizeColorTexture();
        this.#resizeDepthTexture();
        this.#resizeInputBuffer();
        this.#resizeOutputTexture();
    }

    #resizeInputBuffer() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        const size   = 1 * 4 * width * height * 4;
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

    #resizeColorTexture() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.directColorTexture_A?.width  === width &&
            this.directColorTexture_A?.height === height) return;

        this.directColorTexture_A?.destroy();
        this.directColorTexture_A = this.device.createTexture({
            size  : [width, height],
            format: this.splatFormat,
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }

    #resizeDepthTexture() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.directDepthTexture_A?.width  === width &&
            this.directDepthTexture_A?.height === height) return;

        this.directDepthTexture_A?.destroy();
        this.directDepthTexture_A = this.device.createTexture({
            size  : [width, height],
            format: 'depth24plus',
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }
}
import { Camera } from 'engine/core.js';
import { SplatRenderer } from 'renderers/SplatRenderer.js';
import { Compositor } from 'renderers/Compositor.js';
import { DepthCompositor } from 'renderers/DepthCompositor.js';
import { TemporalConfidence } from 'renderers/TemporalConfidence.js';
import { TextureToTensorConfidence } from 'renderers/TextureToTensorConfidence.js'
import { FP32ToFP16Converter } from 'renderers/FP32ToFP16Converter.js';
import { TensorToTextureConverter } from 'renderers/TensorToTextureConverter.js';
import * as ort from 'onnxruntime-web/webgpu';

import { PerformanceTracker } from '../PerformanceTracker.js';

import stochastic_splatting_code from 'shaders/stochastic_splat_render.wgsl?raw';
import temporal_confidence_code from 'shaders/temporal_confidence.wgsl?raw';

export class RenderingPipelineModelInferrenceConfidenceFP16 {
    constructor({
        device, context, format, canvas, scene, camera, onnxModel, performanceTracker
    }) {
        this.device  = device;
        this.context = context;
        this.format  = format;
        this.perf    = performanceTracker ?? new PerformanceTracker();

        this.splatFormat  = 'rgba8unorm';
        this.baseChannels = 24;
        this.pad_factor   = 32;
        this.canvas       = canvas;
        this.scene        = scene;
        this.camera       = camera;

        this.renderer                 = new SplatRenderer(device, stochastic_splatting_code, this.splatFormat, false);
        this.compositor               = new Compositor(device, format);
        this.depth_converter          = new DepthCompositor(device, 'rgba8unorm');
        this.temporal_confidence      = new TemporalConfidence(device, temporal_confidence_code, 'rgba8unorm');
        this.textureToTensorConverter = new TextureToTensorConfidence(device);
        this.fp32ToFp16Converter      = new FP32ToFP16Converter(device);
        this.tensorToTextureConverter = new TensorToTextureConverter(device, 'rgba8unorm', true);

        this.directColorTexture_A    = null;
        this.directDepthTexture_A    = null;
        this.depthExportTexture      = null;
        this.confidenceExportTexture = null;
        this.inferenceInputBufferF32 = null;
        this.inferenceInputBufferF16 = null;
        this.inferenceOutputTexture  = null;

        this.historyColorTexture_A      = null;
        this.historyColorTexture_B      = null;
        this.historyDepthTexture_A      = null;
        this.historyDepthTexture_B      = null;
        this.historyConfidenceTexture_A = null;
        this.historyConfidenceTexture_B = null;
        this.historyPingPongFlip        = false;

        this.inferenceInFlight       = false;
        this.last_render_timestamp   = 0;
        this.last_render_duration_ms = 0;
        this.completed_render_count  = 0;

        this.onnx_model        = onnxModel;
        this.hiddenStates      = {};
        this.hiddenReady       = false;
        this.hiddenInputNames  = onnxModel.session.inputNames.filter(n => /^h\d+_in$/.test(n)).sort();
        this.hiddenOutputNames = this.hiddenInputNames.map(n => n.replace('_in', '_out'));
        this.hiddenDims        = {};
        this.runOptions        = {};
        this.onnxModelReady    = true;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 'r' && !event.repeat) {
                console.log('[Pipeline] reset hidden state');
                this.#resetHidden();
            }
        });

        console.log('[Pipeline] inputMetadata:',  this.onnx_model.session.handler.inputMetadata);
        console.log('[Pipeline] outputMetadata:', this.onnx_model.session.handler.outputMetadata);
    }

    update(_t, _dt) {}

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
        this.#resetHidden();
        this.#cacheHiddenDims(width, height);
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
        const width  = this.canvas.width;
        const height = this.canvas.height;

        this.perf.begin('noisy_render');
        this.renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );
        //await this.device.queue.onSubmittedWorkDone();
        this.perf.end('noisy_render');

        this.perf.begin('confidence');
        const currentHistoryColor      = this.#getCurrentHistoryColorTexture();
        const currentHistoryDepth      = this.#getCurrentHistoryDepthTexture();
        const currentHistoryConfidence = this.#getCurrentHistoryConfidenceTexture();
        const nextHistoryColor         = this.#getNextHistoryColorTexture();
        const nextHistoryDepth         = this.#getNextHistoryDepthTexture();
        const nextHistoryConfidence    = this.#getNextHistoryConfidenceTexture();

        this.temporal_confidence.render(
            {
                confidence:        this.confidenceExportTexture,
                historyColor:      nextHistoryColor,
                historyDepth:      nextHistoryDepth,
                historyConfidence: nextHistoryConfidence,
            },
            this.directColorTexture_A,
            this.directDepthTexture_A,
            this.camera,
            currentHistoryColor,
            currentHistoryDepth,
            currentHistoryConfidence,
        );
        this.#swapTemporalHistoryBuffers();
        //await this.device.queue.onSubmittedWorkDone();
        this.perf.end('confidence');

        this.perf.begin('texture_to_tensor');
        this.textureToTensorConverter.render(
            this.directColorTexture_A,
            this.directDepthTexture_A,
            this.confidenceExportTexture,
            this.inferenceInputBufferF32,
            width, height
        );
        //await this.device.queue.onSubmittedWorkDone();
        this.perf.end('texture_to_tensor');

        this.perf.begin('fp32_to_fp16');
        this.fp32ToFp16Converter.convert(
            this.inferenceInputBufferF32,
            this.inferenceInputBufferF16,
            5 * width * height
        );
        //await this.device.queue.onSubmittedWorkDone();
        this.perf.end('fp32_to_fp16');

        this.perf.begin('inferrence_tensor_prep');
        const inputTensor = new ort.Tensor({
            location : 'gpu-buffer',
            gpuBuffer: this.inferenceInputBufferF16,
            type     : 'float16',
            dims     : [1, 5, height, width],
        });

        const feeds = { input: inputTensor };

        for (const name of this.hiddenInputNames) {
            const key = name.replace('_in', '');
            if (this.hiddenReady && this.hiddenStates[key]) {
                const { buffer, dims } = this.hiddenStates[key];
                feeds[name] = new ort.Tensor({
                    location : 'gpu-buffer',
                    gpuBuffer: buffer,
                    type     : 'float16',
                    dims,
                });
            } else {
                const dims   = this.hiddenDims[name];
                const numel  = dims.reduce((a, b) => a * b, 1);
                feeds[name]  = new ort.Tensor('float16', new Uint16Array(numel), dims);
            }
        }
        this.perf.end('inferrence_tensor_prep');

        this.perf.begin('inferrence_call');
        let results;
        try {
            results = await this.onnx_model.session.run(feeds, this.runOptions);
        } finally {
            inputTensor.dispose?.();
            for (const name of this.hiddenInputNames) feeds[name]?.dispose?.();
        }
        this.perf.end('inferrence_call');

        this.perf.begin('inferrence_sync');
        //await this.device.queue.onSubmittedWorkDone();
        this.perf.end('inferrence_sync');

        this.perf.begin('hidden_state_swap');
        for (const outName of this.hiddenOutputNames) {
            const key    = outName.replace('_out', '');
            const tensor = results[outName];
            if (!tensor?.gpuBuffer)
                throw new Error(`Hidden state output '${outName}' is not GPU-backed`);
            this.hiddenStates[key]?.buffer?.destroy();
            this.hiddenStates[key] = {
                buffer: tensor.gpuBuffer,
                dims  : Array.from(tensor.dims),
            };
        }
        this.hiddenReady = true;
        this.perf.end('hidden_state_swap');

        this.perf.begin('output_tensor_to_texture');
        const outMain = results['output'];
        if (!outMain?.gpuBuffer)
            throw new Error("Main output 'output' is not GPU-backed");
        this.tensorToTextureConverter.render(
            outMain.gpuBuffer,
            this.inferenceOutputTexture,
            width, height
        );
        //await this.device.queue.onSubmittedWorkDone();
        outMain.dispose?.();
        this.perf.end('output_tensor_to_texture');

        this.perf.begin('screen_render');
        this.compositor.render(
            { color: this.context.getCurrentTexture() },
            this.inferenceOutputTexture,
            1.0
        );
        //await this.device.queue.onSubmittedWorkDone();
        this.perf.end('screen_render');
    }

    #cacheHiddenDims(width, height) {
        const pad_factor = this.pad_factor;
        const pH = Math.ceil(height / pad_factor) * pad_factor;
        const pW = Math.ceil(width  / pad_factor) * pad_factor;
        const allMeta = this.onnx_model.session.handler?.inputMetadata;

        for (const name of this.hiddenInputNames) {
            const match = name.match(/^h(\d+)_in$/);
            if (!match) throw new Error(`Cannot infer dims for '${name}'`);
            const level        = parseInt(match[1], 10) - 1;
            const spatialScale = 1 << level;
            const meta = Array.isArray(allMeta) ? allMeta.find(m => m.name === name) : null;
            const channels = (meta?.shape?.[1] && typeof meta.shape[1] === 'number')
                ? meta.shape[1]
                : this.baseChannels * spatialScale;
            this.hiddenDims[name] = [1, channels, pH / spatialScale, pW / spatialScale];
        }
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
        this.#resizeDepthExportTexture();
        this.#resizeConfidenceExportTexture();
        this.#resizeTemporalHistoryTextures();
        this.#resizeInputBuffers();
        this.#resizeOutputTexture();
    }

    #resizeInputBuffers() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        const numel  = 5 * width * height;

        const f32Size = numel * 4;
        const f16Size = numel * 2;

        if (this.inferenceInputBufferF32?.size === f32Size) return;

        this.inferenceInputBufferF32?.destroy();
        this.inferenceInputBufferF16?.destroy();

        this.inferenceInputBufferF32 = this.device.createBuffer({
            size : f32Size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.inferenceInputBufferF16 = this.device.createBuffer({
            size : f16Size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.#resetHidden();
        this.#cacheHiddenDims(width, height);
    }

    #resizeColorTexture() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.directColorTexture_A?.width === width &&
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
        if (this.directDepthTexture_A?.width === width &&
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

    #resizeDepthExportTexture() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.depthExportTexture?.width === width &&
            this.depthExportTexture?.height === height) return;
        this.depthExportTexture?.destroy();
        this.depthExportTexture = this.device.createTexture({
            size  : [width, height],
            format: 'rgba8unorm',
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }

    #resizeConfidenceExportTexture() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.confidenceExportTexture?.width === width &&
            this.confidenceExportTexture?.height === height) return;
        this.confidenceExportTexture?.destroy();
        this.confidenceExportTexture = this.device.createTexture({
            size  : [width, height],
            format: 'rgba8unorm',
            usage :
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING   |
                GPUTextureUsage.COPY_SRC,
        });
    }

    #resizeTemporalHistoryTextures() {
        const width  = this.canvas.width;
        const height = this.canvas.height;
        if (this.historyColorTexture_A?.width === width &&
            this.historyColorTexture_A?.height === height) return;

        this.historyColorTexture_A?.destroy();
        this.historyColorTexture_B?.destroy();
        this.historyDepthTexture_A?.destroy();
        this.historyDepthTexture_B?.destroy();
        this.historyConfidenceTexture_A?.destroy();
        this.historyConfidenceTexture_B?.destroy();

        this.historyColorTexture_A      = this.#createColorHistoryTexture();
        this.historyColorTexture_B      = this.#createColorHistoryTexture();
        this.historyDepthTexture_A      = this.#createDepthHistoryTexture();
        this.historyDepthTexture_B      = this.#createDepthHistoryTexture();
        this.historyConfidenceTexture_A = this.#createConfidenceHistoryTexture();
        this.historyConfidenceTexture_B = this.#createConfidenceHistoryTexture();

        this.historyPingPongFlip = false;
        this.temporal_confidence.reset();
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

    #createColorHistoryTexture() {
        return this.device.createTexture({
            size  : [this.canvas.width, this.canvas.height],
            format: 'rgba8unorm',
            usage : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #createDepthHistoryTexture() {
        return this.device.createTexture({
            size  : [this.canvas.width, this.canvas.height],
            format: 'r32float',
            usage : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #createConfidenceHistoryTexture() {
        return this.device.createTexture({
            size  : [this.canvas.width, this.canvas.height],
            format: 'r32float',
            usage : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #getCurrentHistoryColorTexture()      { return this.historyPingPongFlip ? this.historyColorTexture_B      : this.historyColorTexture_A; }
    #getNextHistoryColorTexture()         { return this.historyPingPongFlip ? this.historyColorTexture_A      : this.historyColorTexture_B; }
    #getCurrentHistoryDepthTexture()      { return this.historyPingPongFlip ? this.historyDepthTexture_B      : this.historyDepthTexture_A; }
    #getNextHistoryDepthTexture()         { return this.historyPingPongFlip ? this.historyDepthTexture_A      : this.historyDepthTexture_B; }
    #getCurrentHistoryConfidenceTexture() { return this.historyPingPongFlip ? this.historyConfidenceTexture_B : this.historyConfidenceTexture_A; }
    #getNextHistoryConfidenceTexture()    { return this.historyPingPongFlip ? this.historyConfidenceTexture_A : this.historyConfidenceTexture_B; }
    #swapTemporalHistoryBuffers()         { this.historyPingPongFlip = !this.historyPingPongFlip; }
}
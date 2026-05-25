import { mat4 } from 'glm';
import { Camera } from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { getGlobalViewMatrix, getProjectionMatrix } from 'engine/core/SceneUtils.js';

import { TextureToTensorConverter } from './TextureToTensorConverter.js';
import { TensorToTextureConverter } from './TensorToTextureConverter.js';
import { ImageSampler } from './ImageSampler.js';
import { DepthCompositor } from './DepthCompositor.js';
import { TemporalConfidence } from './TemporalConfidence.js';
import * as ort from 'onnxruntime-web/webgpu';

const stochastic_splatting_code = await fetch(
    new URL('./shaders/stochastic_splat_render.wgsl', import.meta.url)
).then(r => r.text());

const sorted_splatting_code = await fetch(
    new URL('./shaders/sorted_splat_render.wgsl', import.meta.url)
).then(r => r.text());

const temporal_confidence_code = await fetch(
    new URL('./shaders/temporal_confidence.wgsl', import.meta.url)
).then(r => r.text());

export class RenderingPipelineDatasetGatherConfidence {
    constructor({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        onnxModel
    }) {
        this.device  = device;
        this.context = context;
        this.format  = format;

        this.splatFormat = 'rgba8unorm';

        this.canvas = canvas;
        this.scene  = scene;
        this.camera = camera;

        this.renderer              = new SplatRenderer(device, stochastic_splatting_code, this.splatFormat, false);
        this.ground_truth_renderer = new SplatRenderer(device, sorted_splatting_code,     this.splatFormat, true);
        this.image_sampler         = new ImageSampler(device, 'nike');
        this.depth_converter       = new DepthCompositor(device, 'rgba8unorm');
        this.compositor            = new Compositor(device, format);
        this.temporal_confidence   = new TemporalConfidence(device, temporal_confidence_code, 'rgba8unorm');
        this.textureToTensorConverter = new TextureToTensorConverter(device);
        this.tensorToTextureConverter = new TensorToTextureConverter(device);

        this.directColorTexture_A   = null;
        this.directDepthTexture_A   = null;
        this.depthExportTexture     = null;

        this.historyColorTexture_A      = null;
        this.historyColorTexture_B      = null;
        this.historyDepthTexture_A      = null;
        this.historyDepthTexture_B      = null;
        this.historyConfidenceTexture_A = null;
        this.historyConfidenceTexture_B = null;
        this.confidenceExportTexture    = null;
        this.historyPingPongFlip        = false;

        this.saveRequested            = false;
        this.last_render_timestamp    = 0;
        this.last_render_duration_ms  = 0;
        this.completed_render_count   = 0;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 's')
                this.saveRequested = !this.saveRequested;
        });
    }

    update(_t, _dt) {}

    resize(width, height) {
        console.log("Pipeline Resize Called");
        this.camera.getComponentOfType(Camera).aspect = width / height;
    }

    async render() {
        if (this.trainingRenderInFlight) return false;

        this.trainingRenderInFlight = true;
        const t0 = performance.now();

        this.#ensureTrainingResources();

        // --- noisy render ---
        this.renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );
        this.depth_converter.render(
            { color: this.depthExportTexture },
            this.directDepthTexture_A
        );

        // --- confidence accumulation ---
        const currentHistoryColor      = this.#getCurrentHistoryColorTexture();
        const currentHistoryDepth      = this.#getCurrentHistoryDepthTexture();
        const currentHistoryConfidence = this.#getCurrentHistoryConfidenceTexture();
        const nextHistoryColor         = this.#getNextHistoryColorTexture();
        const nextHistoryDepth         = this.#getNextHistoryDepthTexture();
        const nextHistoryConfidence    = this.#getNextHistoryConfidenceTexture();

        this.temporal_confidence.render(
            {
                historyColor:      nextHistoryColor,
                historyDepth:      nextHistoryDepth,
                historyConfidence: nextHistoryConfidence,
                confidence:        this.confidenceExportTexture,
            },
            this.directColorTexture_A,
            this.directDepthTexture_A,
            this.camera,
            currentHistoryColor,
            currentHistoryDepth,
            currentHistoryConfidence,
        );

        this.#swapTemporalHistoryBuffers();

        let noiseColor, noiseDepth, noiseConfidence;
        if (this.saveRequested) {
            await this.device.queue.onSubmittedWorkDone();
            noiseColor      = await this.image_sampler.readTexturePixels(this.directColorTexture_A);
            noiseDepth      = await this.image_sampler.readTexturePixels(this.depthExportTexture);
            noiseConfidence = await this.image_sampler.readTexturePixels(this.confidenceExportTexture);
        }

        this.ground_truth_renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );
        this.depth_converter.render(
            { color: this.depthExportTexture },
            this.directDepthTexture_A
        );

        const viewMatrix = getGlobalViewMatrix(this.camera);
        const projectionMatrix = getProjectionMatrix(this.camera);
        const currentViewProjectionMatrix = mat4.create();
        mat4.multiply(currentViewProjectionMatrix, projectionMatrix, viewMatrix);

        if (this.saveRequested) {
            await this.device.queue.onSubmittedWorkDone();
            const gtColor = await this.image_sampler.readTexturePixels(this.directColorTexture_A);
            const gtDepth = await this.image_sampler.readTexturePixels(this.depthExportTexture);

            this.image_sampler.queueSave(noiseColor, noiseDepth, 'noise', false);
            this.image_sampler.queueSaveSingle(noiseConfidence, 'confidence', false);
            this.image_sampler.queueSave(gtColor, gtDepth, 'gt', true);
            await this.image_sampler._saveChain;
        }

        this.compositor.render(
            { color: this.context.getCurrentTexture() },
            this.confidenceExportTexture,
            1.0
        );

        await this.device.queue.onSubmittedWorkDone();

        this.last_render_timestamp   = performance.now();
        this.last_render_duration_ms = performance.now() - t0;
        this.completed_render_count++;
        this.trainingRenderInFlight  = false;

        return true;
    }

    #ensureTrainingResources() {
        this.#resizeDirectColorTexture();
        this.#resizeDirectDepthTexture();
        this.#resizeDepthExportTexture();
        this.#resizeTemporalHistoryTextures();
        this.#resizeConfidenceExportTexture();
    }

    #resizeDirectColorTexture() {
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

    #resizeDirectDepthTexture() {
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
            format: 'rgba8unorm',
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
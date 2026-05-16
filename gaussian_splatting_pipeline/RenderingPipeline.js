import { Camera } from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { TemporalDenoiser } from './TemporalDenoiser.js';
// import { SpatialDenoiser } from './SpatialDenoiser.js';


const stochastic_splatting_code = await fetch(new URL('./shaders/stochastic_splat_render.wgsl', import.meta.url)).then(response => response.text());
const sorted_splatting_code = await fetch(new URL('./shaders/sorted_splat_render.wgsl', import.meta.url)).then(response => response.text());

const temporal_denoiser_code = await fetch(new URL('./shaders/temporal_denoising.wgsl', import.meta.url)).then(response => response.text());
const edge_temporal_denoiser_code = await fetch(new URL('./shaders/temporal_edge_denoising.wgsl', import.meta.url)).then(response => response.text());

export class RenderingPipeline {
    constructor({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
    }) {
        this.device = device;
        this.context = context;
        this.format = format;
        this.splatFormat = 'rgba16float';

        this.canvas = canvas;
        this.scene = scene;
        this.camera = camera;

        //this.renderer = new SplatRenderer(device, stochastic_splatting_code, this.splatFormat, false);
        this.renderer = new SplatRenderer(device, sorted_splatting_code, this.splatFormat, true);

        this.temporal_denoiser = new TemporalDenoiser(this.device, temporal_denoiser_code, this.splatFormat);
        this.temporal_edge_denoiser = new TemporalDenoiser(this.device, edge_temporal_denoiser_code, this.splatFormat);

        this.compositor = new Compositor(device, format);
        this.compositor.gamma = 1;

        this.directColorTexture_A = null;
        this.directColorTexture_B = null;
        this.directColorPingPongFlip = false;

        this.directDepthTexture = null;

        this.historyColorTexture_A = null;
        this.historyColorTexture_B = null;

        this.historyDepthTexture_A = null;
        this.historyDepthTexture_B = null;

        this.historyConfidenceTexture_A = null;
        this.historyConfidenceTexture_B = null;

        this.historyPingPongFlip = false;

        this.debugTexture = null;
    }

    update(t, dt) {
        // nothing
    }

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
    }

    render() {
        this.#resizeDirectColorTextures();
        this.#resizeDirectDepthTexture();
        this.#resizeDebugTexture();
        this.#resizeTemporalHistoryTextures();

        this.directColorPingPongFlip = false;

        let current_directColorTexture = this.#getCurrentDirectColorTexture();
        let next_directColorTexture = this.#getNextDirectColorTexture();

        let current_historyColorTexture = this.#getCurrentHistoryColorTexture();
        let current_historyDepthTexture = this.#getCurrentHistoryDepthTexture();
        let current_historyConfidenceTexture = this.#getCurrentHistoryConfidenceTexture();

        let next_historyColorTexture = this.#getNextHistoryColorTexture();
        let next_historyDepthTexture = this.#getNextHistoryDepthTexture();
        let next_historyConfidenceTexture = this.#getNextHistoryConfidenceTexture();

        const splattingRenderTarget = {
            color: current_directColorTexture,
            depth: this.directDepthTexture,
        };

        this.renderer.render(splattingRenderTarget, this.scene, this.camera);

        // main temporal denoiser pass
        const temporalDenoiserRenderTarget = {
            color: next_directColorTexture,

            historyColor: next_historyColorTexture,
            historyDepth: next_historyDepthTexture,
            historyConfidence: next_historyConfidenceTexture,

            debug: this.debugTexture,
        };

        this.temporal_denoiser.render(
            temporalDenoiserRenderTarget,
            current_directColorTexture,
            this.directDepthTexture,
            this.camera,
            current_historyColorTexture,
            current_historyDepthTexture,
            current_historyConfidenceTexture,
        );


        //buffer swap
        this.#swapDirectColorTextures();
        this.#swapTemporalHistoryBuffers();

        current_directColorTexture = this.#getCurrentDirectColorTexture();
        next_directColorTexture = this.#getNextDirectColorTexture();

        current_historyColorTexture = this.#getCurrentHistoryColorTexture();
        current_historyDepthTexture = this.#getCurrentHistoryDepthTexture();
        current_historyConfidenceTexture = this.#getCurrentHistoryConfidenceTexture();

        next_historyColorTexture = this.#getNextHistoryColorTexture();
        next_historyDepthTexture = this.#getNextHistoryDepthTexture();
        next_historyConfidenceTexture = this.#getNextHistoryConfidenceTexture();

        // second temporal denoise pass
        const edgeTemporalDenoiserRenderTarget = {
            color: next_directColorTexture,

            historyColor: next_historyColorTexture,
            historyDepth: next_historyDepthTexture,
            historyConfidence: next_historyConfidenceTexture,

            debug: this.debugTexture,
        };

        this.temporal_edge_denoiser.render(
            edgeTemporalDenoiserRenderTarget,
            current_directColorTexture,
            this.directDepthTexture,
            this.camera,
            current_historyColorTexture,
            current_historyDepthTexture,
            current_historyConfidenceTexture,
        );

        //buffer swap
        this.#swapDirectColorTextures();

        current_directColorTexture = this.#getCurrentDirectColorTexture();

        //canvas render
        const canvasRenderTarget = {
            color: this.context.getCurrentTexture(),
        };

        this.compositor.render(canvasRenderTarget, current_directColorTexture, 1.0);

        // this.compositor.render(canvasRenderTarget, this.debugTexture, 1.0);
    }

    // INTERNAL

    #resizeDirectColorTextures() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        if (
            this.directColorTexture_A &&
            this.directColorTexture_A.width === width &&
            this.directColorTexture_A.height === height
        ) {
            return;
        }

        this.directColorTexture_A?.destroy();
        this.directColorTexture_B?.destroy();

        this.directColorTexture_A = this.#createDirectColorTexture();
        this.directColorTexture_B = this.#createDirectColorTexture();

        this.directColorPingPongFlip = false;
    }

    #resizeDirectDepthTexture() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        if (
            this.directDepthTexture &&
            this.directDepthTexture.width === width &&
            this.directDepthTexture.height === height
        ) {
            return;
        }

        this.directDepthTexture?.destroy();

        this.directDepthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #resizeDebugTexture() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        if (
            this.debugTexture &&
            this.debugTexture.width === width &&
            this.debugTexture.height === height
        ) {
            return;
        }

        this.debugTexture?.destroy();

        this.debugTexture = this.device.createTexture({
            size: [width, height],
            format: this.splatFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #resizeTemporalHistoryTextures() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        if (
            this.historyColorTexture_A &&
            this.historyColorTexture_A.width === width &&
            this.historyColorTexture_A.height === height
        ) {
            return;
        }

        this.historyColorTexture_A?.destroy();
        this.historyColorTexture_B?.destroy();

        this.historyDepthTexture_A?.destroy();
        this.historyDepthTexture_B?.destroy();

        this.historyConfidenceTexture_A?.destroy();
        this.historyConfidenceTexture_B?.destroy();

        this.historyColorTexture_A = this.#createTemporalColorHistoryTexture();
        this.historyColorTexture_B = this.#createTemporalColorHistoryTexture();

        this.historyDepthTexture_A = this.#createTemporalScalarHistoryTexture();
        this.historyDepthTexture_B = this.#createTemporalScalarHistoryTexture();

        this.historyConfidenceTexture_A = this.#createTemporalScalarHistoryTexture();
        this.historyConfidenceTexture_B = this.#createTemporalScalarHistoryTexture();

        this.historyPingPongFlip = false;

        this.temporal_denoiser.reset();
        this.temporal_edge_denoiser.reset();
    }

    #createDirectColorTexture() {
        return this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: this.splatFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #createTemporalColorHistoryTexture() {
        return this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: this.splatFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #createTemporalScalarHistoryTexture() {
        return this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'r32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #getCurrentDirectColorTexture() {
        return this.directColorPingPongFlip
            ? this.directColorTexture_B
            : this.directColorTexture_A;
    }

    #getNextDirectColorTexture() {
        return this.directColorPingPongFlip
            ? this.directColorTexture_A
            : this.directColorTexture_B;
    }

    #swapDirectColorTextures() {
        this.directColorPingPongFlip = !this.directColorPingPongFlip;
    }

    #getCurrentHistoryColorTexture() {
        return this.historyPingPongFlip
            ? this.historyColorTexture_B
            : this.historyColorTexture_A;
    }

    #getNextHistoryColorTexture() {
        return this.historyPingPongFlip
            ? this.historyColorTexture_A
            : this.historyColorTexture_B;
    }

    #getCurrentHistoryDepthTexture() {
        return this.historyPingPongFlip
            ? this.historyDepthTexture_B
            : this.historyDepthTexture_A;
    }

    #getNextHistoryDepthTexture() {
        return this.historyPingPongFlip
            ? this.historyDepthTexture_A
            : this.historyDepthTexture_B;
    }

    #getCurrentHistoryConfidenceTexture() {
        return this.historyPingPongFlip
            ? this.historyConfidenceTexture_B
            : this.historyConfidenceTexture_A;
    }

    #getNextHistoryConfidenceTexture() {
        return this.historyPingPongFlip
            ? this.historyConfidenceTexture_A
            : this.historyConfidenceTexture_B;
    }

    #swapTemporalHistoryBuffers() {
        this.historyPingPongFlip = !this.historyPingPongFlip;
    }
}
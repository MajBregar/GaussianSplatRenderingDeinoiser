import { Camera } from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { TemporalDenoiser } from './TemporalDenoiser.js';
// import { SpatialDenoiser } from './SpatialDenoiser.js';

import { WebGpuDenoiser } from './WebGpuDenoiser.js'
import { TextureToTensorConverter } from './TextureToTensorConverter.js';
import { TensorToTextureConverter } from './TensorToTextureConverter.js';
import * as ort from 'onnxruntime-web/webgpu';


import { ImageSampler } from './ImageSampler.js';
import { DepthCompositor } from './DepthCompositor.js';

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

        this.splatFormat = 'rgba8unorm';

        this.canvas = canvas;
        this.scene = scene;
        this.camera = camera;

        this.renderer = new SplatRenderer(device, stochastic_splatting_code, this.splatFormat, false);
        this.ground_truth_renderer = new SplatRenderer(device, sorted_splatting_code, this.splatFormat, true);

        this.temporal_denoiser = new TemporalDenoiser(this.device, temporal_denoiser_code, this.splatFormat);
        this.temporal_edge_denoiser = new TemporalDenoiser(this.device, edge_temporal_denoiser_code, this.splatFormat);

        this.image_sampler = new ImageSampler(this.device, './samples_1');
        this.depth_converter = new DepthCompositor(this.device, 'rgba8unorm')
        this.depthExportTexture = null;



        this.compositor = new Compositor(device, format);
        this.compositor.gamma = 1;

        this.directColorTexture_A = null;
        this.directColorTexture_B = null;
        this.directColorPingPongFlip = false;

        this.directDepthTexture_A = null;
        this.directDepthTexture_B = null;

        this.historyColorTexture_A = null;
        this.historyColorTexture_B = null;

        this.historyDepthTexture_A = null;
        this.historyDepthTexture_B = null;

        this.historyConfidenceTexture_A = null;
        this.historyConfidenceTexture_B = null;

        this.historyPingPongFlip = false;

        this.debugTexture = null;


        this.saveRequested = false;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 's' && !event.repeat) {
                this.saveRequested = true;
            }
        });

        //testing tensor computation
        this.inferenceInputBuffer = null;
        this.inferenceOutputBuffer = null;
        this.inferenceOutputTexture = null;

        this.textureToTensorConverter = new TextureToTensorConverter(this.device);
        this.tensorToTextureConverter = new TensorToTextureConverter(this.device);

        this.webgpu_denoiser = new WebGpuDenoiser('/models/tiny_denoiser.onnx');
        this.webgpuDenoiserReady = false;

        this.webgpu_denoiser.init(device).then(() => {
            this.webgpuDenoiserReady = true;
            this.inferenceInputBuffer?.destroy();
            this.inferenceInputBuffer = null;

            this.inferenceOutputBuffer?.destroy();
            this.inferenceOutputBuffer = null;
        });

        console.log(ort);
        console.log(this.webgpu_denoiser.session);

        this.inferenceInFlight = false;
        this.inferenceOutputReady = false;

    }

    update(t, dt) {
        // nothing
    }

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
    }

    render_old_OLD_DEPTH_TEX() {
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

    train_set_render() {
        this.#resizeDirectColorTextures();
        this.#resizeDirectDepthTexture();
        this.#resizeDepthExportTexture();


        const noisy_splat_render_target = {
            color: this.directColorTexture_A,
            depth: this.directDepthTexture_A,
        };
        this.renderer.render(noisy_splat_render_target, this.scene, this.camera);

        const gt_splat_render_target = {
            color: this.directColorTexture_B,
            depth: this.directDepthTexture_B,
        };
        this.ground_truth_renderer.render(gt_splat_render_target, this.scene, this.camera);

        const depth_convert_target = {
            color: this.depthExportTexture,
        };
        this.depth_converter.render(depth_convert_target, this.directDepthTexture_B);

        if (this.saveRequested) {
            this.saveRequested = false;
            this.image_sampler.savePair(this.directColorTexture_B, this.depthExportTexture);
        }

        //canvas render
        const canvasRenderTarget = {
            color: this.context.getCurrentTexture(),
        };

        this.compositor.render(canvasRenderTarget, this.directColorTexture_B, 1.0);
    }


    async inferrence_render() {
        this.#resizeDirectColorTextures();
        this.#resizeDirectDepthTexture();
        this.#resizeInferenceBuffers();

        const width = this.canvas.width;
        const height = this.canvas.height;

        this.renderer.render(
            {
                color: this.directColorTexture_A,
                depth: this.directDepthTexture_A,
            },
            this.scene,
            this.camera
        );

        if (!this.webgpuDenoiserReady || this.inferenceInFlight) {
            this.compositor.render(
                { color: this.context.getCurrentTexture() },
                this.inferenceOutputReady ? this.inferenceOutputTexture : this.directColorTexture_A,
                1.0
            );
            return;
        }

        this.inferenceInFlight = true;

        let inputTensor = null;
        let outputTensor = null;

        try {
            this.textureToTensorConverter.render(
                this.directColorTexture_A,
                this.directDepthTexture_A,
                this.inferenceInputBuffer,
                width,
                height
            );

            const inputName = this.webgpu_denoiser.session.inputNames[0];
            const outputName = this.webgpu_denoiser.session.outputNames[0];

            inputTensor = new ort.Tensor({
                location: 'gpu-buffer',
                gpuBuffer: this.inferenceInputBuffer,
                type: 'float32',
                dims: [1, 4, height, width],
            });

            const results = await this.webgpu_denoiser.session.run(
                {
                    [inputName]: inputTensor,
                },
                {
                    preferredOutputLocation: {
                        [outputName]: 'gpu-buffer',
                    },
                }
            );

            outputTensor = results[outputName];

            if (outputTensor.location !== 'gpu-buffer') {
                console.warn('ONNX output is CPU-backed.');
                return;
            }

            this.tensorToTextureConverter.render(
                outputTensor.gpuBuffer,
                this.inferenceOutputTexture,
                width,
                height
            );

            await this.device.queue.onSubmittedWorkDone();

            this.inferenceOutputReady = true;

            this.compositor.render(
                { color: this.context.getCurrentTexture() },
                this.inferenceOutputTexture,
                1.0
            );
        } catch (error) {
            console.error('ONNX inference failed:', error?.message ?? error);
        } finally {
            inputTensor?.dispose?.();
            outputTensor?.dispose?.();

            this.inferenceInFlight = false;
        }
    }


    // INTERNAL
    #resizeInferenceBuffers() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        //console.log(width, height);

        const inputSizeBytes = 1 * 4 * width * height * 4;
        const outputSizeBytes = 1 * 3 * width * height * 4;

        if (
            this.inferenceInputBuffer &&
            this.inferenceInputBuffer.size === inputSizeBytes &&
            this.inferenceOutputBuffer.size === outputSizeBytes
        ) {
            return;
        }

        this.inferenceInputBuffer?.destroy();
        this.inferenceOutputBuffer?.destroy();
        this.inferenceOutputTexture?.destroy();

        this.inferenceInputBuffer = this.device.createBuffer({
            size: inputSizeBytes,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        });

        this.inferenceOutputBuffer = this.device.createBuffer({
            size: outputSizeBytes,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        });

        this.inferenceOutputTexture = this.device.createTexture({
            size: [width, height],
            format: this.splatFormat,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });
    }


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
            this.directDepthTexture_A &&
            this.directDepthTexture_A.width === width &&
            this.directDepthTexture_A.height === height
        ) {
            return;
        }

        this.directDepthTexture_A?.destroy();
        this.directDepthTexture_B?.destroy();

        this.directDepthTexture_A = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });

        this.directDepthTexture_B = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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

    #resizeDepthExportTexture() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        if (
            this.depthExportTexture &&
            this.depthExportTexture.width === width &&
            this.depthExportTexture.height === height
        ) {
            return;
        }

        this.depthExportTexture?.destroy();

        this.depthExportTexture = this.device.createTexture({
            size: [width, height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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
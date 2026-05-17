import { Camera } from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';

import { OnnxModelInitializer } from './OnnxModelInitializer.js';
import { TextureToTensorConverter } from './TextureToTensorConverter.js';
import { TensorToTextureConverter } from './TensorToTextureConverter.js';
import * as ort from 'onnxruntime-web/webgpu';

import { ImageSampler } from './ImageSampler.js';
import { DepthCompositor } from './DepthCompositor.js';

const stochastic_splatting_code = await fetch(
    new URL('./shaders/stochastic_splat_render.wgsl', import.meta.url)
).then(response => response.text());

const sorted_splatting_code = await fetch(
    new URL('./shaders/sorted_splat_render.wgsl', import.meta.url)
).then(response => response.text());

export class RenderingPipeline {
    constructor({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        modelName
    }) {
        this.device = device;
        this.context = context;
        this.format = format;

        this.splatFormat = 'rgba8unorm';

        this.canvas = canvas;
        this.scene = scene;
        this.camera = camera;

        this.renderer = new SplatRenderer(
            device,
            stochastic_splatting_code,
            this.splatFormat,
            false
        );

        this.ground_truth_renderer = new SplatRenderer(
            device,
            sorted_splatting_code,
            this.splatFormat,
            true
        );

        this.image_sampler = new ImageSampler(this.device, './samples_1');
        this.depth_converter = new DepthCompositor(this.device, 'rgba8unorm');
        this.compositor = new Compositor(device, format);

        this.directColorTexture_A = null;
        this.directColorTexture_B = null;

        this.directDepthTexture_A = null;
        this.directDepthTexture_B = null;

        this.depthExportTexture = null;
        this.debugTexture = null;

        this.inferenceInputBuffer = null;
        this.inferenceOutputTexture = null;

        this.textureToTensorConverter = new TextureToTensorConverter(this.device);
        this.tensorToTextureConverter = new TensorToTextureConverter(this.device);

        this.onnx_model = new OnnxModelInitializer(modelName);
        this.onnxModelReady = false;

        this.inputName = null;
        this.outputName = null;
        this.runOptions = null;

        this.inferenceInFlight = false;

        this.saveRequested = false;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 's' && !event.repeat) {
                this.saveRequested = true;
            }
        });

        this.onnx_model.init().then(() => {
            this.inputName = this.onnx_model.session.inputNames[0];
            this.outputName = this.onnx_model.session.outputNames[0];

            this.runOptions = {
                preferredOutputLocation: {
                    [this.outputName]: 'gpu-buffer',
                },
            };

            this.inferenceInputBuffer?.destroy();
            this.inferenceInputBuffer = null;

            this.inferenceOutputTexture?.destroy();
            this.inferenceOutputTexture = null;

            this.onnxModelReady = true;
        });

    }

    update(t, dt) {
        // nothing
    }

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
    }

    async train_set_render() {
        this.#ensureTrainingResources();

        this.renderer.render(
            {
                color: this.directColorTexture_A,
                depth: this.directDepthTexture_A,
            },
            this.scene,
            this.camera
        );

        this.ground_truth_renderer.render(
            {
                color: this.directColorTexture_B,
                depth: this.directDepthTexture_B,
            },
            this.scene,
            this.camera
        );

        this.depth_converter.render(
            {
                color: this.depthExportTexture,
            },
            this.directDepthTexture_B
        );

        if (this.saveRequested) {
            this.saveRequested = false;
            this.image_sampler.savePair(
                this.directColorTexture_B,
                this.depthExportTexture
            );
        }

        this.compositor.render(
            {
                color: this.context.getCurrentTexture(),
            },
            this.directColorTexture_B,
            1.0
        );

        await this.device.queue.onSubmittedWorkDone();
    }

    async inferrence_render() {
        if (!this.onnxModelReady || this.inferenceInFlight) {
            return;
        }

        this.inferenceInFlight = true;

        try {
            await this.#renderInferenceFrameLocked();
        } catch (error) {
            console.error('ONNX inference failed:', error?.message ?? error);
        } finally {
            await this.device.queue.onSubmittedWorkDone();
            this.inferenceInFlight = false;
        }
    }

    async #renderInferenceFrameLocked() {
        this.#ensureInferenceResources();

        const width = this.canvas.width;
        const height = this.canvas.height;

        let inputTensor = null;
        let outputTensor = null;

        // 1. Render scene into intermediate textures.
        this.renderer.render(
            {
                color: this.directColorTexture_A,
                depth: this.directDepthTexture_A,
            },
            this.scene,
            this.camera
        );

        await this.device.queue.onSubmittedWorkDone();

        // 2. Convert color + depth textures into ONNX input buffer.
        this.textureToTensorConverter.render(
            this.directColorTexture_A,
            this.directDepthTexture_A,
            this.inferenceInputBuffer,
            width,
            height
        );

        await this.device.queue.onSubmittedWorkDone();

        try {
            // 3. Wrap GPU input buffer as ONNX tensor.
            inputTensor = new ort.Tensor({
                location: 'gpu-buffer',
                gpuBuffer: this.inferenceInputBuffer,
                type: 'float32',
                dims: [1, 4, height, width],
            });

            // 4. Run ONNX inference.
            const results = await this.onnx_model.session.run(
                {
                    [this.inputName]: inputTensor,
                },
                this.runOptions
            );

            outputTensor = results[this.outputName];

            if (!outputTensor) {
                throw new Error(`Missing ONNX output '${this.outputName}'.`);
            }

            if (outputTensor.location !== 'gpu-buffer' || !outputTensor.gpuBuffer) {
                console.error('Invalid ONNX output tensor:', outputTensor);
                throw new Error('ONNX output is not GPU-backed.');
            }

            await this.device.queue.onSubmittedWorkDone();

            // 5. Convert ONNX output buffer into texture.
            this.tensorToTextureConverter.render(
                outputTensor.gpuBuffer,
                this.inferenceOutputTexture,
                width,
                height
            );

            await this.device.queue.onSubmittedWorkDone();

            // 6. Composite final denoised texture to canvas.
            this.compositor.render(
                {
                    color: this.context.getCurrentTexture(),
                },
                this.inferenceOutputTexture,
                1.0
            );

            await this.device.queue.onSubmittedWorkDone();
        } finally {
            inputTensor?.dispose?.();
            outputTensor?.dispose?.();
        }
    }

    #ensureTrainingResources() {
        this.#resizeDirectColorTextures();
        this.#resizeDirectDepthTexture();
        this.#resizeDepthExportTexture();
    }

    #ensureInferenceResources() {
        this.#resizeDirectColorTextures();
        this.#resizeDirectDepthTexture();
        this.#resizeInferenceBuffers();
    }

    #resizeInferenceBuffers() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        const inputSizeBytes = 1 * 4 * width * height * 4;

        if (
            this.inferenceInputBuffer &&
            this.inferenceInputBuffer.size === inputSizeBytes &&
            this.inferenceOutputTexture &&
            this.inferenceOutputTexture.width === width &&
            this.inferenceOutputTexture.height === height
        ) {
            return;
        }

        this.inferenceInputBuffer?.destroy();
        this.inferenceOutputTexture?.destroy();

        this.inferenceInputBuffer = this.device.createBuffer({
            size: inputSizeBytes,
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

        this.directColorTexture_A = this.device.createTexture({
            size: [width, height],
            format: this.splatFormat,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        this.directColorTexture_B = this.device.createTexture({
            size: [width, height],
            format: this.splatFormat,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });
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
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        this.directDepthTexture_B = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
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
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
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
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        });
    }
}
import { Camera } from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';

import { OnnxModelInitializer } from './OnnxModelInitializer.js';
import { TextureToTensorConverter } from './TextureToTensorConverter.js';
import { TensorToTextureConverter } from './TensorToTextureConverter.js';
import * as ort from 'onnxruntime-web/webgpu';

const stochastic_splatting_code = await fetch(
    new URL('./shaders/stochastic_splat_render.wgsl', import.meta.url)
).then(response => response.text());

export class RenderingPipelineTemporalInferrence {
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
        this.inferenceOutputTexture = null;
        this.inferenceInputBuffer   = null;

         this.hiddenBuffers = {
            h1: null,
            h2: null,
            h3: null,
            h4: null, 
        };
        this.hiddenDims = {
            h1: null, h2: null, h3: null, h4: null,
        };
        this.hiddenReady = false;

        this.onnx_model     = onnxModel;
        this.onnxModelReady = false;
        this.runOptions     = null;

        this.inferenceInFlight       = false;
        this.last_render_timestamp   = 0;
        this.last_render_duration_ms = 0;
        this.completed_render_count  = 0;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 'r' && !event.repeat) {
                console.log('[Pipeline] R → reset hidden state');
                this.#resetHidden();
            }
        });

        this.onnx_model.init().then(() => {
            console.log('[Pipeline] ONNX model ready');
            // console.log('[Pipeline] inputs :', this.onnx_model.session.inputNames);
            // console.log('[Pipeline] outputs:', this.onnx_model.session.outputNames);
 
            this.runOptions = { };

            this.onnxModelReady = true;
        });
    }


    update(_t, _dt) {}

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
        this.#resetHidden();
    }

    async render() {
        if (!this.onnxModelReady) {
            return false;
        }
        if (this.inferenceInFlight) {
            return false;
        }

        this.inferenceInFlight = true;
        const t0 = performance.now();

        try {
            await this.#renderFrame();
            const dt = performance.now() - t0;
            this.last_render_timestamp   = performance.now();
            this.last_render_duration_ms = dt;
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

        this.renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );

        this.textureToTensorConverter.render(
            this.directColorTexture_A,
            this.directDepthTexture_A,
            this.inferenceInputBuffer,
            width, height
        );

        await this.device.queue.onSubmittedWorkDone();

        const base = this.baseChannels;
        const h2w  = Math.floor(width  / 2), h2h = Math.floor(height / 2);
        const h4w  = Math.floor(width  / 4), h4h = Math.floor(height / 4);
        const h8w  = Math.floor(width  / 8), h8h = Math.floor(height / 8);

        const inputTensor = new ort.Tensor({
            location : 'gpu-buffer',
            gpuBuffer: this.inferenceInputBuffer,
            type     : 'float32',
            dims     : [1, 4, height, width],
        });

        let h1in, h2in, h3in, h4in;

        if (this.hiddenReady) {
            h1in = new ort.Tensor({ location: 'gpu-buffer', gpuBuffer: this.hiddenBuffers.h1, type: 'float32', dims: this.hiddenDims.h1 });
            h2in = new ort.Tensor({ location: 'gpu-buffer', gpuBuffer: this.hiddenBuffers.h2, type: 'float32', dims: this.hiddenDims.h2 });
            h3in = new ort.Tensor({ location: 'gpu-buffer', gpuBuffer: this.hiddenBuffers.h3, type: 'float32', dims: this.hiddenDims.h3 });
            h4in = new ort.Tensor({ location: 'gpu-buffer', gpuBuffer: this.hiddenBuffers.h4, type: 'float32', dims: this.hiddenDims.h4 });
        } else {
            const z = (dims) => new ort.Tensor('float32', new Float32Array(dims.reduce((a,b)=>a*b,1)), dims);
            h1in = z([1, base,     height, width]);
            h2in = z([1, base * 2, h2h,    h2w  ]);
            h3in = z([1, base * 4, h4h,    h4w  ]);
            h4in = z([1, base * 8, h8h,    h8w  ]);
        }

        const feeds = {
            'input': inputTensor,
            'h1_in': h1in,
            'h2_in': h2in,
            'h3_in': h3in,
            'h4_in': h4in,
        };

        let results;
        try {
            results = await this.onnx_model.session.run(feeds, this.runOptions);
        } finally {
            inputTensor.dispose?.();
            h1in.dispose?.();
            h2in.dispose?.();
            h3in.dispose?.();
            h4in.dispose?.();
        }

        await this.device.queue.onSubmittedWorkDone();

        const outH1 = results['h1_out'];
        const outH2 = results['h2_out'];
        const outH3 = results['h3_out'];
        const outH4 = results['h4_out'];

        if (!outH1?.gpuBuffer || !outH2?.gpuBuffer || !outH3?.gpuBuffer || !outH4?.gpuBuffer)
            throw new Error('Hidden state outputs are not GPU-backed');

        if (!this.hiddenReady) {
            for (const [key, tensor, dimKey] of [
                ['h1', outH1, 'h1'], ['h2', outH2, 'h2'],
                ['h3', outH3, 'h3'], ['h4', outH4, 'h4'],
            ]) {
                this.hiddenBuffers[key]?.destroy();
                this.hiddenBuffers[key] = this.device.createBuffer({
                    size : outH1.gpuBuffer.size,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                });
                this.hiddenDims[key] = Array.from(tensor.dims);
            }

            this.hiddenBuffers.h1?.destroy();
            this.hiddenBuffers.h2?.destroy();
            this.hiddenBuffers.h3?.destroy();
            this.hiddenBuffers.h4?.destroy();
            this.hiddenBuffers.h1 = this.#mkBuf(outH1.gpuBuffer.size);
            this.hiddenBuffers.h2 = this.#mkBuf(outH2.gpuBuffer.size);
            this.hiddenBuffers.h3 = this.#mkBuf(outH3.gpuBuffer.size);
            this.hiddenBuffers.h4 = this.#mkBuf(outH4.gpuBuffer.size);
            this.hiddenDims.h1 = Array.from(outH1.dims);
            this.hiddenDims.h2 = Array.from(outH2.dims);
            this.hiddenDims.h3 = Array.from(outH3.dims);
            this.hiddenDims.h4 = Array.from(outH4.dims);
            this.hiddenReady = true;
        }

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(outH1.gpuBuffer, 0, this.hiddenBuffers.h1, 0, Math.min(outH1.gpuBuffer.size, this.hiddenBuffers.h1.size));
        enc.copyBufferToBuffer(outH2.gpuBuffer, 0, this.hiddenBuffers.h2, 0, Math.min(outH2.gpuBuffer.size, this.hiddenBuffers.h2.size));
        enc.copyBufferToBuffer(outH3.gpuBuffer, 0, this.hiddenBuffers.h3, 0, Math.min(outH3.gpuBuffer.size, this.hiddenBuffers.h3.size));
        enc.copyBufferToBuffer(outH4.gpuBuffer, 0, this.hiddenBuffers.h4, 0, Math.min(outH4.gpuBuffer.size, this.hiddenBuffers.h4.size));
        this.device.queue.submit([enc.finish()]);

        outH1.dispose?.();
        outH2.dispose?.();
        outH3.dispose?.();
        outH4.dispose?.();

        await this.device.queue.onSubmittedWorkDone();

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

        this.compositor.render(
            { color: this.context.getCurrentTexture() },
            this.inferenceOutputTexture,
            1.0
        );
        await this.device.queue.onSubmittedWorkDone();
    }


    #mkBuf(size) {
        return this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    #resetHidden() {
        this.hiddenBuffers.h1?.destroy();
        this.hiddenBuffers.h2?.destroy();
        this.hiddenBuffers.h3?.destroy();
        this.hiddenBuffers.h4?.destroy();
        this.hiddenBuffers = { h1: null, h2: null, h3: null, h4: null };
        this.hiddenDims    = { h1: null, h2: null, h3: null, h4: null };
        this.hiddenReady   = false;
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
        if (this.inferenceOutputTexture?.width === width &&
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
}
import { mat4 } from 'glm';
import { Camera } from 'engine/core.js';
import { SplatRenderer } from 'renderers/SplatRenderer.js';
import { Compositor } from 'renderers/Compositor.js';
import { getGlobalViewMatrix, getProjectionMatrix } from 'engine/core/SceneUtils.js';

import { TextureToTensorConverter } from 'renderers/TextureToTensorConverter.js';
import { TensorToTextureConverter } from 'renderers/TensorToTextureConverter.js';
import { DepthCompositor } from 'renderers/DepthCompositor.js';
import * as ort from 'onnxruntime-web/webgpu';

import { ImageSampler } from '../ImageSampler.js';

import stochastic_splatting_code from 'shaders/stochastic_splat_render.wgsl?raw';
import sorted_splatting_code from 'shaders/sorted_splat_render.wgsl?raw';

export class RenderingPipelineDatasetGather {
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
        this.textureToTensorConverter = new TextureToTensorConverter(device);
        this.tensorToTextureConverter = new TensorToTextureConverter(device);

        this.directColorTexture_A   = null;
        this.directDepthTexture_A   = null;
        this.depthExportTexture     = null;

        this.saveRequested            = false;
        this.last_render_timestamp    = 0;
        this.last_render_duration_ms  = 0;
        this.completed_render_count   = 0;

        window.addEventListener('keydown', event => {
            if (event.key.toLowerCase() === 's')
                this.saveRequested = true;
        });
        window.addEventListener('keyup', event => {
            if (event.key.toLowerCase() === 's')
                this.saveRequested = false;
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


        this.renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );
        this.depth_converter.render(
            { color: this.depthExportTexture },
            this.directDepthTexture_A
        );

        let noiseColor, noiseDepth;
        if (this.saveRequested) {
            await this.device.queue.onSubmittedWorkDone();
            noiseColor = await this.image_sampler.readTexturePixels(this.directColorTexture_A);
            noiseDepth = await this.image_sampler.readTexturePixels(this.depthExportTexture);
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
            //this.image_sampler.queueSave(noiseColor, noiseDepth, 'noise', false);
            this.image_sampler.queueSaveWithCamera(noiseColor, noiseDepth, 'noise', currentViewProjectionMatrix, false);
            this.image_sampler.queueSave(gtColor, gtDepth, 'gt', true);
            await this.image_sampler._saveChain; //stall pipeline until saved
        }

        this.compositor.render(
            { color: this.context.getCurrentTexture() },
            this.directColorTexture_A,
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
}
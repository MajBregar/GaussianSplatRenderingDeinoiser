import { mat4 } from 'glm';
import { Camera } from 'engine/core.js';
import { SplatRenderer } from 'renderers/SplatRenderer.js';
import { Compositor } from 'renderers/Compositor.js';
import { getGlobalViewMatrix, getProjectionMatrix } from 'engine/core/SceneUtils.js';
import { MotionMapCompositor } from 'renderers/MotionMapCompositor.js';
import { DepthCompositor } from 'renderers/DepthCompositor.js';

import stochastic_splatting_code from 'shaders/stochastic_splat_render.wgsl?raw';
import sorted_splatting_code from 'shaders/sorted_splat_render.wgsl?raw';

import { ImageSampler } from '../ImageSampler.js';

export class RenderingPipelineDatasetGatherMotionMap {
    constructor({ device, context, format, canvas, scene, camera }) {
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
        this.compositor            = new Compositor(device, format);
        this.motionMapCompositor   = new MotionMapCompositor(device, 'rgba8unorm');
        this.depth_converter       = new DepthCompositor(device, 'rgba8unorm');


        this.directColorTexture_A = null;
        this.directDepthTexture_A = null;
        this.motionMapTexture     = null;
        this.depthExportTexture     = null;


        this.previousViewProjectionMatrix = null;

        this.saveRequested           = false;
        this.last_render_timestamp   = 0;
        this.last_render_duration_ms = 0;
        this.completed_render_count  = 0;

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
        if (this.renderInFlight) return false;

        this.renderInFlight = true;
        const t0 = performance.now();

        this.#ensureResources();

        const viewMatrix       = getGlobalViewMatrix(this.camera);
        const projectionMatrix = getProjectionMatrix(this.camera);
        const currentVP        = mat4.create();
        mat4.multiply(currentVP, projectionMatrix, viewMatrix);

        const previousVP = this.previousViewProjectionMatrix ?? mat4.clone(currentVP);

        this.renderer.render(
            { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
            this.scene, this.camera
        );

        this.motionMapCompositor.render(
            { color: this.motionMapTexture },
            this.directDepthTexture_A,
            currentVP,
            previousVP
        );

        await this.device.queue.onSubmittedWorkDone();

        this.previousViewProjectionMatrix = mat4.clone(currentVP);

        if (this.saveRequested) {
            

            const noiseColor  = await this.image_sampler.readTexturePixels(this.directColorTexture_A);
            const motionMap   = await this.image_sampler.readTexturePixels(this.motionMapTexture);

            this.depth_converter.render(
                { color: this.depthExportTexture },
                this.directDepthTexture_A
            );
            await this.device.queue.onSubmittedWorkDone();

            this.ground_truth_renderer.render(
                { color: this.directColorTexture_A, depth: this.directDepthTexture_A },
                this.scene, this.camera
            );
            
            await this.device.queue.onSubmittedWorkDone();

            const noiseDepth   = await this.image_sampler.readTexturePixels(this.depthExportTexture);
            const gtColor = await this.image_sampler.readTexturePixels(this.directColorTexture_A);

            this.image_sampler.queueSave(noiseColor, noiseDepth, 'noise', false);
            this.image_sampler.queueSaveSingle(motionMap, 'motion', false);
            this.image_sampler.queueSaveSingle(gtColor, 'gt_color', true);
            await this.image_sampler._saveChain;
        }

        this.compositor.render(
            { color: this.context.getCurrentTexture() },
            this.motionMapTexture,
            1.0
        );

        await this.device.queue.onSubmittedWorkDone();

        this.last_render_timestamp   = performance.now();
        this.last_render_duration_ms = performance.now() - t0;
        this.completed_render_count++;
        this.renderInFlight = false;

        return true;
    }

    #ensureResources() {
        this.#resizeDirectColorTexture();
        this.#resizeDirectDepthTexture();
        this.#resizeMotionMapTexture();
        this.#resizeDepthExportTexture();
    }

    #resizeDirectColorTexture() {
        const { width, height } = this.canvas;
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
        const { width, height } = this.canvas;
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

    #resizeMotionMapTexture() {
        const { width, height } = this.canvas;
        if (this.motionMapTexture?.width === width &&
            this.motionMapTexture?.height === height) return;

        this.motionMapTexture?.destroy();
        this.motionMapTexture = this.device.createTexture({
            size  : [width, height],
            format: 'rgba8unorm',
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
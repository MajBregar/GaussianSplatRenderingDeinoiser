import {Camera, Node, Transform} from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';

//import { Denoiser } from './Denoiser_passthrough.js';
import { TemporalDenoiser } from './TemporalDenoiser.js';
import { DepthCompositor } from './DepthCompositor.js';
import { SpatialDenoiser } from './SpatialDenoiser.js';

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

        this.renderer = new SplatRenderer(device, this.splatFormat);

        this.temporal_denoiser = new TemporalDenoiser(this.device, this.splatFormat);

        this.spatial_denoiser = new SpatialDenoiser(this.device, this.splatFormat);

        this.compositor = new Compositor(device, format);
        this.compositor.gamma = 1;

        //this.debug_depth_compositor = new DepthCompositor(device, format);

        //BUFFERS / TEXTURES
        this.depthTexture = null;
        this.colorTexture = null;
        this.compositorTexture = null;
        this.spatialTexture = null;

        // STATE VARIABLES
    }

    // CALLABLE

    instantResetHandler() {
    }

    update(t, dt) {
        // nothing
    }

    resize(width, height) {
        this.camera.getComponentOfType(Camera).aspect = width / height;
        this.instantResetHandler();
    }

    render() {
        this.#resizeColorTexture();
        this.#resizeDepthTexture();
        this.#resizeCompositorTexture();
        this.#resizeSpatialTexture();
        this.temporal_denoiser.resize(this.canvas.width, this.canvas.height);
        this.spatial_denoiser.resize(this.canvas.width, this.canvas.height);

        
        // Gaussian splat rendering
        const splatting_render_target = {
            color: this.colorTexture,
            depth: this.depthTexture,
        };
        this.renderer.render(splatting_render_target, this.scene, this.camera);
        

        // // Denoising
        const temporal_denoiser_render_target = {
            color: this.compositorTexture,
        }
        this.temporal_denoiser.render(temporal_denoiser_render_target, this.colorTexture, this.depthTexture, this.camera);
        

        const spatial_denoiser_render_target = {
            color: this.spatialTexture,
        };
        this.spatial_denoiser.maxConfidence = this.temporal_denoiser.maxHistoryConfidence;

        this.spatial_denoiser.render(
            spatial_denoiser_render_target,
            this.compositorTexture,
            this.depthTexture,
            this.temporal_denoiser.historyConfidenceTexture
        );

        const canvas_render_target = {
            color: this.context.getCurrentTexture(),
        };
        this.compositor.render(canvas_render_target, this.spatialTexture, 1.0);

        // this.debug_depth_compositor.render(canvas_render_target, this.depthTexture, 1.0);
    }


    // INTERNAL
    
    #resizeColorTexture() {
        if (this.colorTexture && this.colorTexture.width === this.canvas.width &&  this.colorTexture.height === this.canvas.height) {
            return;
        }

        this.colorTexture?.destroy();
        this.colorTexture = this.device.createTexture({
            size: [
                this.canvas.width,
                this.canvas.height,
            ],
            format: this.splatFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #resizeDepthTexture() {
        if (this.depthTexture && this.depthTexture.width === this.canvas.width && this.depthTexture.height === this.canvas.height) {
            return;
        }

        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            size: [
                this.canvas.width,
                this.canvas.height,
            ],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #resizeCompositorTexture() {
        if (this.compositorTexture && this.compositorTexture.width === this.canvas.width && this.compositorTexture.height === this.canvas.height) {
            return;
        }

        this.compositorTexture?.destroy();
        this.compositorTexture = this.device.createTexture({
            size: [
                this.canvas.width,
                this.canvas.height,
            ],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    #resizeSpatialTexture() {
        if (this.spatialTexture && this.spatialTexture.width === this.canvas.width && this.spatialTexture.height === this.canvas.height) {
            return;
        }

        this.spatialTexture?.destroy();

        this.spatialTexture = this.device.createTexture({
            size: [
                this.canvas.width,
                this.canvas.height,
            ],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }
}
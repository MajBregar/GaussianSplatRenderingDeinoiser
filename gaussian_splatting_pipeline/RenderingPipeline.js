import {Camera, Node, Transform} from 'engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { Denoiser } from './Denoiser_passthrough.js';

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

        this.canvas = canvas;
        this.scene = scene;
        this.camera = camera;

        this.renderer = new SplatRenderer(device, format);

        this.denoiser = new Denoiser(device, 'rgba32float');

        this.compositor = new Compositor(device, format);
        this.compositor.gamma = 1;

        //BUFFERS / TEXTURES
        this.depthTexture = null;
        this.colorTexture = null;
        this.compositorTexture = null;

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

        
        // Gaussian splat rendering
        const splatting_render_target = {
            color: this.colorTexture,
            depth: this.depthTexture,
        };
        this.renderer.render(splatting_render_target, this.scene, this.camera);
        

        // Denoising
        const denoiser_render_target = {
            color: this.compositorTexture,
        }
        this.denoiser.render(denoiser_render_target, this.colorTexture, this.depthTexture);


        // Output compositorTexture to canvas and gamma correct / alpha blend with prev texture (alpha = 1 is fully replace)
        const canvas_render_target = {
            color: this.context.getCurrentTexture(),
        };
        this.compositor.render(canvas_render_target, this.compositorTexture, 1.0);
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
            format: this.format,
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
            format: 'rgba32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }
}
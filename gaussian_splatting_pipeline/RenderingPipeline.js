import {Camera, Node, Transform} from './engine/core.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';

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

        this.compositorFloat = new Compositor(device, 'rgba32float');

        this.compositor = new Compositor(device, format);
        this.compositor.gamma = 1;

        //BUFFERS / TEXTURES
        this.depthTexture = null;
        this.colorTexture = null;
        this.compositorTexture = null;

        // STATE VARIABLES
        this.nFrames = 0;
    }

    // CALLABLE

    instantResetHandler() {
        this.#resetAccumulation()
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

        const renderTarget = {
            color: this.colorTexture,
            depth: this.depthTexture,
        };

        const compositorTarget = {
            color: this.compositorTexture,
        };

        const canvasTarget = {
            color: this.context.getCurrentTexture(),
        };

        this.renderer.render(renderTarget, this.scene, this.camera);

        this.nFrames += 1;
        this.compositorFloat.render(compositorTarget, this.colorTexture, 1 / this.nFrames, 1);

        this.compositor.render(canvasTarget,this.compositorTexture);
    }


    // INTERNAL
    
    #resetAccumulation() {
        this.nFrames = 0;
    }

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
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
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
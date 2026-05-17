import { GUI } from 'dat';
import { vec3, mat4 } from 'glm';

import { ResizeSystem } from 'engine/systems/ResizeSystem.js';
import { UpdateSystem } from 'engine/systems/UpdateSystem.js';
import {Camera, Node, Transform} from 'engine/core.js';
import { TouchController } from 'engine/controllers/TouchController.js';

import { parseSplats } from './gaussian_splatting_pipeline/file_handling/parseSplats.js';
import { Splat } from './gaussian_splatting_pipeline/file_handling/Splat.js';

import { RenderingPipeline } from './gaussian_splatting_pipeline/RenderingPipeline.js';
import { SplatLoader } from './gaussian_splatting_pipeline/file_handling/SplatLoader.js';

import { OnnxModelInitializer } from './gaussian_splatting_pipeline/OnnxModelInitializer.js';
import * as ort from 'onnxruntime-web/webgpu';

const MODEL_NAME = '/models/tiny_denoiser.onnx';
const INFERENCE_WIDTH = 1280;
const INFERENCE_HEIGHT = 720;

// webbpu init
const onnxModel = new OnnxModelInitializer(MODEL_NAME);
await onnxModel.init();

const device = onnxModel.device;
const onnxSession = onnxModel.session;

const canvas = document.querySelector('canvas');
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({device, format});

canvas.width = INFERENCE_WIDTH;
canvas.height = INFERENCE_HEIGHT;
canvas.style.width = `${INFERENCE_WIDTH}px`;
canvas.style.height = `${INFERENCE_HEIGHT}px`;

context.configure({device,format});

// scene
const scene = new Node();
const splatContainer = new Node();
scene.addChild(splatContainer);

const camera = new Node();
camera.addComponent(new Transform());
camera.addComponent(new Camera());
camera.addComponent(new TouchController(camera, canvas));
scene.addChild(camera);


//rendering setup
const renderingPipeline = new RenderingPipeline({
    device,
    context,
    format,
    canvas,
    scene,
    camera,
});

const splatLoader = new SplatLoader({
    canvas,
    splatContainer,
    renderingPipeline,
    defaultFile: './splats/nike.splat',
});

await splatLoader.initialize();
splatLoader.enableDragAndDrop();


camera.addComponent({
    lastTransform: mat4.clone(camera.getComponentOfType(Transform).matrix),

    update() {
        const transform = camera.getComponentOfType(Transform);

        if (mat4.exactEquals(transform.matrix, this.lastTransform)) {
            return;
        }

        mat4.copy(this.lastTransform, transform.matrix);
    }
});



// GUI
function makeGUIControllerReadOnly(controller) {
    const input = controller.domElement.querySelector('input');
    if (input) {
        input.readOnly = true;
        input.disabled = true;
        input.style.pointerEvents = 'none';
    }
    controller.domElement.style.pointerEvents = 'none';
    controller.domElement.style.opacity = '0.75';
}

const UPDATE_FRAME_STATS_EVERY_N_SECONDS = 1.0;

const performance_stats = {
    fps: '0',
    frame_time: '0.00',
};

let frame_counter = 0;
let runtime_counter = 0;

const gui = new GUI();

const fpsController = gui.add(performance_stats, 'fps').name('FPS').listen();
const frameTimeController = gui.add(performance_stats, 'frame_time').name('Frame time ms').listen();
makeGUIControllerReadOnly(fpsController);
makeGUIControllerReadOnly(frameTimeController);

// gui.add(renderingPipeline.renderer, 'splatScale', 0, 10).name("Splat Scale");
// gui.add(renderingPipeline.renderer, 'loBound', 0, 1).name("Lower Bound");
// gui.add(renderingPipeline.renderer, 'hiBound', 0, 1).name("Higher Bound");
// gui.add(renderingPipeline.compositor, 'gamma', 0, 3).name("Gamma Correction");

// gui.add(renderingPipeline.temporal_denoiser, 'historyWeight', 0, 1).name("TAA History Weight");
// gui.add(renderingPipeline.temporal_denoiser, 'depthThreshold', 0, 0.1).name("TAA Depth Thr");

// gui.add(renderingPipeline.temporal_denoiser, 'maxHistoryConfidence', 0, 100).name("maxHistoryConfidence");
// gui.add(renderingPipeline.temporal_denoiser, 'varianceClipGamma', 0, 8).name("varianceClipGamma");
// gui.add(renderingPipeline.temporal_denoiser, 'reprojectionDistanceScale', 0, 100).name("reprojectionDistanceScale");


// gui.add(renderingPipeline.spatial_denoiser, 'depthSigma', 0.0001, 0.1).step(0.0001).name("SD Depth Sigma");
// gui.add(renderingPipeline.spatial_denoiser, 'colorSigma', 0.0, 2.0).step(0.01).name("SD Color Sigma");
// gui.add(renderingPipeline.spatial_denoiser, 'maxConfidence', 1.0, 64.0).step(1.0).name("SD Max Confidence");
// gui.add(renderingPipeline.spatial_denoiser, 'baseStrength', 0.0, 1.0).step(0.01).name("SD Base Strength");
// gui.add(renderingPipeline.spatial_denoiser, 'minSpatialStrength', 0.0, 1.0).step(0.01).name("SD Min Strength");
// gui.add(renderingPipeline.spatial_denoiser, 'fireflyStrength', 0.0, 1.0).step(0.01).name("SD Firefly Strength");

// gui.add(renderingPipeline.debug_depth_compositor, 'depthMin', 0, 1.0).name("DC Depth Min");
// gui.add(renderingPipeline.debug_depth_compositor, 'depthMax', 0, 1.0).name("DC Depth Max");
// gui.add(renderingPipeline.debug_depth_compositor, 'contrast', 0, 1.0).name("DC Contrast");



// render loop wrappers
function update(t, dt) {
    frame_counter++;
    runtime_counter += dt;

    if (runtime_counter >= UPDATE_FRAME_STATS_EVERY_N_SECONDS) {
        performance_stats.fps = Math.round(frame_counter / runtime_counter).toString();
        performance_stats.frame_time = ((runtime_counter / frame_counter) * 1000).toFixed(2);

        frame_counter = 0;
        runtime_counter = 0;
    }

    scene.traverse(node => {
        for (const component of node.components) {
            component.update?.(t, dt);
        }
    });

    renderingPipeline.update(t, dt);
}

function render() {
    renderingPipeline.inferrence_render();
}

function resize({ displaySize: { width, height }}) {
    //renderingPipeline.resize(width, height);
}



// START RENDER
new ResizeSystem({canvas, resize}).start();
new UpdateSystem({update, render}).start();
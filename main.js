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

const GATHER_TRAINING_EXAMPLES = false;
const MODEL_NAME = '/models/SimpleAutoencoder720p_depth.onnx';
const INFERENCE_WIDTH = 1280;
const INFERENCE_HEIGHT = 720;

// webbpu init
const webgpuDenoiser = new OnnxModelInitializer(MODEL_NAME);
await webgpuDenoiser.init();

const device = webgpuDenoiser.device;
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
    modelName: MODEL_NAME
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

const guiState = {
    outputFolder: 'None',
};


const guiActions = {
    selectOutputFolder: async () => {
        try {
            const handle = await renderingPipeline.image_sampler.selectOutputFolder();

            if (handle) {
                guiState.outputFolder = handle.name;
            } else {
                guiState.outputFolder = 'None';
            }
        } catch (error) {
            if (error?.name === 'AbortError') {
                console.log('Output folder selection cancelled.');
                return;
            }

            console.error('Failed to select output folder:', error);
        }
    },
};


gui.add(renderingPipeline.image_sampler, 'sample_limit', 1, 1000).name('Sample Limit').listen();
const sampleCountController = gui.add(renderingPipeline.image_sampler, 'counter').name("Sampled Frames").listen();
makeGUIControllerReadOnly(sampleCountController);

gui.add(guiState, 'outputFolder').name('Sample Output folder').listen();
gui.add(guiActions, 'selectOutputFolder').name('Select output folder');

// gui.add(renderingPipeline.renderer, 'splatScale', 0, 10).name("Splat Scale");
// gui.add(renderingPipeline.renderer, 'loBound', 0, 1).name("Lower Bound");
// gui.add(renderingPipeline.renderer, 'hiBound', 0, 1).name("Higher Bound");
// gui.add(renderingPipeline.compositor, 'gamma', 0, 3).name("Gamma Correction");


// render loop wrappers
function update(t, dt) {
    scene.traverse(node => {
        for (const component of node.components) {
            component.update?.(t, dt);
        }
    });

    renderingPipeline.update(t, dt);
}

async function render() {
    const frameStart = performance.now();

    if (GATHER_TRAINING_EXAMPLES) {
        await renderingPipeline.train_set_render();
    } else {
        await renderingPipeline.inferrence_render();
    }

    const frameEnd = performance.now();

    frame_counter++;
    runtime_counter += (frameEnd - frameStart) / 1000;

    if (runtime_counter >= UPDATE_FRAME_STATS_EVERY_N_SECONDS) {
        performance_stats.fps = Math.round(frame_counter / runtime_counter).toString();

        performance_stats.frame_time = ((runtime_counter / frame_counter) * 1000).toFixed(2);

        frame_counter = 0;
        runtime_counter = 0;
    }
}

function resize({ displaySize: { width, height }}) {
    //renderingPipeline.resize(width, height);
}



// START RENDER
new ResizeSystem({canvas, resize}).start();
new UpdateSystem({update, render}).start();
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

// webbpu init
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({requiredFeatures: ['float32-blendable']});

const canvas = document.querySelector('canvas');
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
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
        renderingPipeline.instantResetHandler();
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

gui.add(renderingPipeline.renderer, 'splatScale', 0, 10).name("Splat Scale");
gui.add(renderingPipeline.renderer, 'loBound', 0, 1).name("Lower Bound");
gui.add(renderingPipeline.renderer, 'hiBound', 0, 1).name("Higher Bound");
gui.add(renderingPipeline.compositor, 'gamma', 0, 3).name("Gamma Correction");


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
    renderingPipeline.render();
}

function resize({ displaySize: { width, height }}) {
    renderingPipeline.resize(width, height);
}







// START RENDER

new ResizeSystem({canvas, resize}).start();
new UpdateSystem({update, render}).start();
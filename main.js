import { GUI } from 'dat';
import { vec3, mat4 } from 'glm';

import { ResizeSystem } from 'engine/systems/ResizeSystem.js';
import { UpdateSystem } from 'engine/systems/UpdateSystem.js';
import {Camera, Node, Transform} from 'engine/core.js';
import { TouchController } from 'engine/controllers/TouchController.js';
import { OrbitController } from 'engine/controllers/OrbitController.js';
import { AutomaticController } from 'engine/controllers/AutomaticController.js';

import { parseSplats } from 'file_handling/parseSplats.js';
import { Splat } from 'file_handling/Splat.js';
import { SplatLoader } from 'file_handling/SplatLoader.js';

import { RenderingPipelineDatasetGather } from 'pipelines/RenderingPipelineDatasetGather.js';
import { RenderingPipelineDatasetGatherConfidence } from 'pipelines/RenderingPipelineDatasetGatherConfidence.js';

import { RenderingPipelineModelInference } from 'pipelines/RenderingPipelineModelInference.js';
import { RenderingPipelineModelInferenceConfidence } from 'pipelines/RenderingPipelineModelInferenceConfidence.js';
import { RenderingPipelineModelInferenceConfidenceFP16} from 'pipelines/RenderingPipelineModelInferenceConfidenceFP16.js';


import { OnnxModelInitializer } from './gaussian_splatting_pipeline/OnnxModelInitializer.js';

import * as ort from 'onnxruntime-web/webgpu';
import { PerformanceTracker } from './gaussian_splatting_pipeline/PerformanceTracker.js';

const USE_CONFIDENCE = true;
const FP_16_MODE = true;
const DATASET_SAMPLE_MODE = false;

const ENABLE_FREEZING = false;
const FREEZE_EVERY_N_FRAMES = 9;
const FREEZE_DURATION_FRAMES = 6;
const LONG_FREEZE_EVERY_N_FREEZES = 10;
const LONG_FREEZE_DURATION_FRAMES = 20;

const INFERENCE_WIDTH = 1280;
const INFERENCE_HEIGHT = 720;

let MODEL_NAME;
if (USE_CONFIDENCE && !FP_16_MODE) {
    MODEL_NAME = '/models/RecurrentDenoisingAutoencoderConfidence_C24_ClosedRooms.onnx'
} else if (USE_CONFIDENCE && FP_16_MODE) {
    MODEL_NAME = '/models/RecurrentDenoisingAutoencoderConfidence_C24_FP16_ClosedRooms.onnx'
} else {
    MODEL_NAME = '/models/RecurrentDenoisingAutoencoder_C24_ClosedRooms.onnx'
}

// webbpu init
const onnxModel = new OnnxModelInitializer(MODEL_NAME);
await onnxModel.init();

const device = onnxModel.device;
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

const start_camera_transform = new Transform({
    translation: [0, 0, 0],
});


let camera_controller;
if (DATASET_SAMPLE_MODE) {
    camera_controller = new AutomaticController(camera, canvas, {
        rotationRate: [0.0000, 0.0005, 0.0000],
        angles: [20, 0, 180],
        distance: 1,
        distanceRate: 0.00000,
        target: [0, 0, 0]
    })
} else {
    camera_controller = new TouchController(camera, canvas, {
        translation: [0, 0, 0],
        rotation: [0, 0, 1, 0],
        distance: 0.7
    });
}

camera.addComponent(start_camera_transform);
camera.addComponent(new Camera());
camera.addComponent(camera_controller);
scene.addChild(camera);


//rendering setup
const performanceTracker = new PerformanceTracker();
let renderingPipeline;

if (DATASET_SAMPLE_MODE) {
    renderingPipeline = new RenderingPipelineDatasetGatherConfidence({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        onnxModel,
        performanceTracker
    });
} else if (USE_CONFIDENCE && !DATASET_SAMPLE_MODE && !FP_16_MODE) {
    renderingPipeline = new RenderingPipelineModelInferenceConfidence({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        onnxModel,
        performanceTracker
    });
} else if (USE_CONFIDENCE && !DATASET_SAMPLE_MODE && FP_16_MODE) {
    renderingPipeline = new RenderingPipelineModelInferenceConfidenceFP16({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        onnxModel,
        performanceTracker
    });
} else if (!USE_CONFIDENCE && !DATASET_SAMPLE_MODE) {
    renderingPipeline = new RenderingPipelineModelInference({
        device,
        context,
        format,
        canvas,
        scene,
        camera,
        onnxModel,
        performanceTracker
    });
} 

const splatLoader = new SplatLoader({
    canvas,
    splatContainer,
    renderingPipeline,
    defaultFile: './splats/office.splat',
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


if (renderingPipeline.image_sampler) {
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


    gui.add(renderingPipeline.image_sampler, 'sample_limit', 1, 2000).name('Sample Limit').listen();
    const sampleCountController = gui.add(renderingPipeline.image_sampler, 'counter').name("Sampled Frames").listen();
    makeGUIControllerReadOnly(sampleCountController);

    gui.add(guiState, 'outputFolder').name('Sample Output folder').listen();
    gui.add(guiActions, 'selectOutputFolder').name('Select output folder');
}

// gui.add(renderingPipeline.renderer, 'splatScale', 0, 10).name("Splat Scale");
// gui.add(renderingPipeline.renderer, 'loBound', 0, 1).name("Lower Bound");
// gui.add(renderingPipeline.renderer, 'hiBound', 0, 1).name("Higher Bound");
// gui.add(renderingPipeline.compositor, 'gamma', 0, 3).name("Gamma Correction");

if (renderingPipeline.temporal_confidence) {
    gui.add(renderingPipeline.temporal_confidence, 'historyWeight', 0, 1).name("historyWeight");
    gui.add(renderingPipeline.temporal_confidence, 'maxHistoryConfidence', 1, 100).name("maxHistoryConfidence");
    gui.add(renderingPipeline.temporal_confidence, 'relativeDepthThreshold', 0, 1).name("relativeDepthThreshold");
    gui.add(renderingPipeline.temporal_confidence, 'reprojectionDistancePixels', 0, 250).name("reprojectionDistancePixels");

    gui.add(renderingPipeline.temporal_confidence, 'colorHistLower', 0, 1).name("colorHistLower");
    gui.add(renderingPipeline.temporal_confidence, 'colorHistUpper', 0, 1).name("colorHistUpper");
}

const style = document.createElement('style');
style.textContent = `
    .dg select {
        background-color: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #3a3a3a;
        border-radius: 3px;
        padding: 0px 4px;
        height: 18px;
        font-size: 11px;
    }
    .dg select option {
        background-color: #1a1a1a;
        color: #e8e8e8;
    }
`;
document.head.appendChild(style);

if ('renderMode' in renderingPipeline) {
    gui.add(renderingPipeline, 'renderMode', { Denoised: 0, Noisy: 1 }).name("Output Mode");
}



// render loop wrappers
function update(t, dt) {
    scene.traverse(node => {
        for (const component of node.components) {
            component.update?.(t, dt);
        }
    });

    renderingPipeline.update(t, dt);
}

const perf_log = [];

function downloadPerfCSV() {
    if (perf_log.length === 0) return;
    const headers = Object.keys(perf_log[0]);
    const rows = perf_log.map(row => headers.map(h => row[h]).join(','));
    const csv = [headers.join(','), ...rows].join('\n');

    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'perf.csv';
    a.click();
}

window.addEventListener('keydown', e => {
    if (e.key.toLowerCase() === 'p') downloadPerfCSV();
});

let trueFrameCounter = 0;
let movingFrameCounter = 0;
let isFrozen = false;
let freezeFramesRemaining = 0;
let freezeCount = 0;

async function render() {
    const completed = await renderingPipeline.render();
    if (!completed) return;

    frame_counter++;
    trueFrameCounter++;
    runtime_counter += renderingPipeline.last_render_duration_ms / 1000;

    if (!isFrozen) {
        movingFrameCounter++;
        if (movingFrameCounter % FREEZE_EVERY_N_FRAMES === 0 && ENABLE_FREEZING) {
            isFrozen = true;
            freezeCount++;
            const isLongFreeze = freezeCount % LONG_FREEZE_EVERY_N_FREEZES === 0;
            freezeFramesRemaining = isLongFreeze ? LONG_FREEZE_DURATION_FRAMES : FREEZE_DURATION_FRAMES;
            camera_controller.pause();
        }
    } else {
        freezeFramesRemaining--;
        if (freezeFramesRemaining <= 0) {
            isFrozen = false;
            camera_controller.resume();
        }
    }

    if (runtime_counter >= UPDATE_FRAME_STATS_EVERY_N_SECONDS) {
        performance_stats.fps = Math.round(frame_counter / runtime_counter).toString();
        performance_stats.frame_time = ((runtime_counter / frame_counter) * 1000).toFixed(2);
        frame_counter = 0;
        runtime_counter = 0;

        if (renderingPipeline.perf){
            const stats = renderingPipeline.perf.summary();
            perf_log.push({ fps: performance_stats.fps, frame_time: performance_stats.frame_time, ...stats });
        }
    }
}

function resize({ displaySize: { width, height }}) {
    renderingPipeline.resize(width, height);
}

// START RENDER
new ResizeSystem({canvas, resize}).start();
new UpdateSystem({update, render}).start();
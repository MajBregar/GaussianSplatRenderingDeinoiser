// main.js

import { GUI } from 'dat';
import { vec3, mat4 } from 'glm';

import { ResizeSystem } from 'engine/systems/ResizeSystem.js';
import { UpdateSystem } from 'engine/systems/UpdateSystem.js';

import {Camera, Node, Transform} from 'engine/core.js';
import { TouchController } from 'engine/controllers/TouchController.js';

import { parseSplats } from './parseSplats.js';
import { Splat } from './Splat.js';

import { RenderingPipeline } from './RenderingPipeline.js';
import { SplatLoader } from './SplatLoader.js';

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




// render loop wrappers
function update(t, dt) {
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

const gui = new GUI();
gui.add(renderingPipeline.renderer, 'splatScale', 0, 10);
gui.add(renderingPipeline.renderer, 'loBound', 0, 1);
gui.add(renderingPipeline.renderer, 'hiBound', 0, 1);
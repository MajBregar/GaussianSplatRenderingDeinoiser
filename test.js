import { OnnxModelInitializer } from './gaussian_splatting_pipeline/OnnxModelInitializer.js';
import * as ort from 'onnxruntime-web/webgpu';
import { PerformanceTracker } from './gaussian_splatting_pipeline/PerformanceTracker.js';

const MODEL_NAME = '/models/RecurrentDenoisingAutoencoder.onnx';
const onnxModel = new OnnxModelInitializer(MODEL_NAME);
await onnxModel.init();

const WIDTH  = 1280;
const HEIGHT = 720;
const PAD_FACTOR  = 16;
const BASE_CHANNELS = 24;

const perf = new PerformanceTracker();

const hiddenInputNames  = onnxModel.session.inputNames.filter(n => /^h\d+_in$/.test(n)).sort();
const hiddenOutputNames = hiddenInputNames.map(n => n.replace('_in', '_out'));

console.log('Hidden inputs :', hiddenInputNames);
console.log('Hidden outputs:', hiddenOutputNames);
console.log('All inputs    :', onnxModel.session.inputNames);
console.log('All outputs   :', onnxModel.session.outputNames);

function inferHiddenDims(inputName, width, height) {
    const pH = Math.ceil(height / PAD_FACTOR) * PAD_FACTOR;
    const pW = Math.ceil(width  / PAD_FACTOR) * PAD_FACTOR;

    const match = inputName.match(/^h(\d+)_in$/);
    if (!match) throw new Error(`Cannot infer dims for '${inputName}'`);

    const level = parseInt(match[1], 10) - 1;
    const spatialScale = 1 << level;

    const allMeta = onnxModel.session.handler?.inputMetadata;
    const meta = Array.isArray(allMeta)
        ? allMeta.find(m => m.name === inputName)
        : null;

    const channels = (meta?.shape?.[1] && typeof meta.shape[1] === 'number')
        ? meta.shape[1]
        : BASE_CHANNELS * spatialScale;

    return [1, channels, pH / spatialScale, pW / spatialScale];
}

let hiddenStates = {};
let hiddenReady  = false;
let iteration    = 0;

const timings = [];

while (true) {
    iteration++;

    const inputData = new Float32Array(1 * 4 * WIDTH * HEIGHT);
    const inputTensor = new ort.Tensor('float32', inputData, [1, 4, HEIGHT, WIDTH]);

    const feeds = { input: inputTensor };

    for (const name of hiddenInputNames) {
        const key = name.replace('_in', '');
        if (hiddenReady && hiddenStates[key]) {
            feeds[name] = hiddenStates[key];
        } else {
            const dims = inferHiddenDims(name, WIDTH, HEIGHT);
            feeds[name] = new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
        }
    }

    const t0 = performance.now();
    let results;
    try {
        results = await onnxModel.session.run(feeds);
    } finally {
        inputTensor.dispose?.();
        for (const name of hiddenInputNames) {
            if (!hiddenReady) feeds[name]?.dispose?.();
        }
    }
    const elapsed = performance.now() - t0;

    timings.push(elapsed);
    const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
    const min = Math.min(...timings);
    const max = Math.max(...timings);

    console.log(`[iter ${iteration}] inference: ${elapsed.toFixed(2)}ms | avg: ${avg.toFixed(2)}ms | min: ${min.toFixed(2)}ms | max: ${max.toFixed(2)}ms`);

    for (const outName of hiddenOutputNames) {
        const key = outName.replace('_out', '');
        hiddenStates[key]?.dispose?.();
        hiddenStates[key] = results[outName];
    }
    hiddenReady = true;

    results['output']?.dispose?.();
}
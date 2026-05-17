import * as ort from 'onnxruntime-web/webgpu';

const wasmUrl = new URL(
    '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm',
    import.meta.url
).href;

const mjsUrl = new URL(
    '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs',
    import.meta.url
).href;

export class OnnxModelInitializer {

    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
    }

    async init() {
        ort.env.wasm.wasmPaths = {
            wasm: wasmUrl,
            mjs: mjsUrl,
        };

        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['webgpu'],
            preferredOutputLocation: 'gpu-buffer',
        });

        this.device = ort.env.webgpu.device;

        if (!this.device) {
            throw new Error('ONNX Runtime did not expose a WebGPU device.');
        }
    }

}
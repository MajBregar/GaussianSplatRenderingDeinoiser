import * as ort from 'onnxruntime-web/webgpu';

const wasmUrl = new URL(
    '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm',
    import.meta.url
).href;

const mjsUrl = new URL(
    '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs',
    import.meta.url
).href;

export class WebGpuDenoiser {
    constructor(modelPath = '/models/tiny_denoiser.onnx') {
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
            enableGraphCapture: true,
        });

        this.device = ort.env.webgpu.device;

        if (!this.device) {
            throw new Error('ONNX Runtime did not expose a WebGPU device.');
        }
    }

    async run(inputFloat32, width, height) {
        const inputTensor = new ort.Tensor(
            'float32',
            inputFloat32,
            [1, 4, height, width]
        );

        const results = await this.session.run({
            [this.session.inputNames[0]]: inputTensor,
        });

        return results[this.session.outputNames[0]].data;
    }
}
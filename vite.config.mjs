import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { viteStaticCopy } from 'vite-plugin-static-copy';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
    resolve: {
        alias: [
            {
                find: /^engine\/(.*)$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/engine/$1'),
            },
            {
                find: /^pipelines\/(.*)$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/pipelines/$1'),
            },
            {
                find: /^renderers\/(.*)$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/renderers/$1'),
            },
            {
                find: /^shaders\/(.*)$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/shaders/$1'),
            },
            {
                find: /^file_handling\/(.*)$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/file_handling/$1'),
            },
            {
                find: /^glm$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/lib/glm.js'),
            },
            {
                find: /^dat$/,
                replacement: path.resolve(__dirname, 'gaussian_splatting_pipeline/lib/dat.js'),
            },
        ],
    },

    assetsInclude: ['**/*.wgsl', '**/*.splat', '**/*.onnx', '**/*.data'],

    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: 'node_modules/onnxruntime-web/dist/*',
                    dest: 'ort',
                },
            ],
        }),
    ],
});
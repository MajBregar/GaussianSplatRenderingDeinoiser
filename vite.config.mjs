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
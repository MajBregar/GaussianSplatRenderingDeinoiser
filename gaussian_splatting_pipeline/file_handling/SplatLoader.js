import { vec3 } from '../lib/glm.js';

import { Node } from '../engine/core.js';

import { parseSplats } from './parseSplats.js';
import { Splat } from './Splat.js';

export class SplatLoader {

    constructor({
        canvas,
        splatContainer,
        renderingPipeline,
        defaultFile = null,
    }) {

        this.canvas = canvas;
        this.splatContainer = splatContainer;
        this.renderingPipeline = renderingPipeline;

        this.defaultFile = defaultFile;
    }

    async initialize() {
        if (this.defaultFile) {
            const loaded = await this.#tryLoadDefaultFile();
            if (loaded) {
                return;
            }
        }
        //console.log('Waiting for drag and drop...');
    }

    enableDragAndDrop() {
        this.canvas.addEventListener('dragover', e => {
            e.preventDefault();
        });

        this.canvas.addEventListener('drop', async e => {
            e.preventDefault();
            const arrayBuffers = await Promise.all(
                [...e.dataTransfer.files].map(
                    file => file.arrayBuffer()
                )
            );
            await this.loadFromArrayBuffers(arrayBuffers);
        });
    }

    async loadFromArrayBuffers(arrayBuffers) {
        this.#clearSplats();

        for (const arrayBuffer of arrayBuffers) {
            const splatData = parseSplats(arrayBuffer);

            this.#centerSplats(splatData);

            const splat = new Node();
            splat.addComponent(new Splat(splatData));

            this.splatContainer.addChild(splat);
        }

        this.renderingPipeline.instantResetHandler();
    }

    async #tryLoadDefaultFile() {
        try {
            //console.log(`Attempting to load ${this.defaultFile}`);

            const response = await fetch(this.defaultFile);

            if (!response.ok) {
                throw new Error('File not found');
            }

            const arrayBuffer = await response.arrayBuffer();

            await this.loadFromArrayBuffers([arrayBuffer]);
            //console.log('Loaded local splat file');

            return true;
        } catch (err) {
            //console.log('No local splat file found');
            return false;
        }
    }

    #clearSplats() {
        for (const child of this.splatContainer.children) {
            child.remove();
        }
    }

    #centerSplats(splatData) {
        const splatMean = splatData
            .map(splat => splat.position)
            .reduce(
                (a, p) => vec3.add(a, a, vec3.scale(vec3.create(), p, 1 / splatData.length)), vec3.create()
            );

        for (const splat of splatData) {
            vec3.subtract(splat.position, splat.position, splatMean);
        }
    }
}
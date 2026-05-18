export class ImageSampler {
    constructor(device, file_prefix) {
        this.device      = device;
        this.counter     = 0;
        this.file_prefix = file_prefix;
        this.outputDirectoryHandle  = null;
        this.sample_limit           = 200;
        this.sampleLimitWarningShown = false;
        this._saveChain  = Promise.resolve();
    }

    async selectOutputFolder() {
        if (!window.showDirectoryPicker)
            throw new Error('File System Access API is not supported in this browser.');

        this.outputDirectoryHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
        return this.outputDirectoryHandle;
    }

    getId() {
        return String(this.counter).padStart(6, '0');
    }

    async save(colorTexture, depthColorTexture) {
        if (this.counter >= this.sample_limit) {
            if (!this.sampleLimitWarningShown) {
                alert(`Sample limit reached (${this.sample_limit}).`);
                this.sampleLimitWarningShown = true;
            }
            return;
        }
        await this.savePair(colorTexture, depthColorTexture);
    }

    async readTexturePixels(texture) {
        return this.#readTexturePixels(texture);
    }

    queueSave(colorData, depthData, image_name, incrementAfter = false) {
        this._saveChain = this._saveChain.then(async () => {
            if (!this.outputDirectoryHandle) return;
            if (this.counter > this.sample_limit - 1) return;
            const id = this.getId();
            await this.#writePngToSelectedFolder(colorData,
                `${this.file_prefix}_${image_name}_color_${id}.png`);
            await this.#writePngToSelectedFolder(depthData,
                `${this.file_prefix}_${image_name}_depth_${id}.png`);
            if (incrementAfter) this.counter++;
        }).catch(e => {
            console.log('Disk Write Busy - Skipping Frame');
        });
    }

    async #readTexturePixels(texture) {
        const width  = texture.width;
        const height = texture.height;
        const unpaddedBytesPerRow = width * 4;
        const bytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;

        const readBuffer = this.device.createBuffer({
            size:  bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture },
            { buffer: readBuffer, bytesPerRow, rowsPerImage: height },
            [width, height, 1]
        );
        this.device.queue.submit([encoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);

        const mapped = new Uint8Array(readBuffer.getMappedRange());
        const pixels = new Uint8ClampedArray(width * height * 4);
        for (let y = 0; y < height; y++) {
            pixels.set(
                mapped.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow),
                y * unpaddedBytesPerRow
            );
        }

        readBuffer.unmap();
        readBuffer.destroy();

        return { pixels, width, height };
    }

    async #writePngToSelectedFolder({ pixels, width, height }, filename) {
        const canvas = document.createElement('canvas');
        canvas.width  = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        ctx.putImageData(new ImageData(pixels, width, height), 0, 0);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        if (!blob) throw new Error('Failed to encode PNG blob.');

        const fileHandle = await this.outputDirectoryHandle.getFileHandle(filename, { create: true });
        const writable   = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
    }
}
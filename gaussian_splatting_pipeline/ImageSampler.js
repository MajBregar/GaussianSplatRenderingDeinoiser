export class ImageSampler {
    constructor(device, outputFolderName = 'samples') {
        this.device = device;
        this.outputFolderName = outputFolderName;
        this.counter = 0;
        this.outputDirectoryHandle = null;
    }

    async selectOutputFolder() {
        if (!window.showDirectoryPicker) {
            throw new Error('File System Access API is not supported in this browser.');
        }

        this.outputDirectoryHandle = await window.showDirectoryPicker({
            mode: 'readwrite',
        });

        return this.outputDirectoryHandle;
    }

    nextId() {
        const id = String(this.counter).padStart(6, '0');
        this.counter++;
        return id;
    }

    async save(colorTexture, depthColorTexture) {
        await this.savePair(colorTexture, depthColorTexture);
    }

    async savePair(colorTexture, depthColorTexture) {
        if (!this.outputDirectoryHandle) {
            await this.selectOutputFolder();
        }

        const id = this.nextId();

        await this.saveColor(colorTexture, `color_${id}.png`);
        await this.saveColor(depthColorTexture, `depth_${id}.png`);
    }

    async saveColor(texture, filename) {
        if (!this.outputDirectoryHandle) {
            await this.selectOutputFolder();
        }

        await this.#saveTextureAsPng(texture, filename);
    }

    async #saveTextureAsPng(texture, filename) {
        const width = texture.width;
        const height = texture.height;

        const bytesPerPixel = 4;
        const unpaddedBytesPerRow = width * bytesPerPixel;
        const bytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
        const bufferSize = bytesPerRow * height;

        const readBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const commandEncoder = this.device.createCommandEncoder();

        commandEncoder.copyTextureToBuffer(
            { texture },
            {
                buffer: readBuffer,
                bytesPerRow,
                rowsPerImage: height,
            },
            [width, height, 1]
        );

        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);

        const mapped = new Uint8Array(readBuffer.getMappedRange());
        const pixels = new Uint8ClampedArray(width * height * 4);

        for (let y = 0; y < height; y++) {
            const srcOffset = y * bytesPerRow;
            const dstOffset = y * unpaddedBytesPerRow;

            pixels.set(
                mapped.subarray(srcOffset, srcOffset + unpaddedBytesPerRow),
                dstOffset
            );
        }

        readBuffer.unmap();
        readBuffer.destroy();

        await this.#writePngToSelectedFolder(pixels, width, height, filename);
    }

    async #writePngToSelectedFolder(pixels, width, height, filename) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(pixels, width, height);

        ctx.putImageData(imageData, 0, 0);

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/png');
        });

        if (!blob) {
            throw new Error('Failed to encode PNG blob.');
        }

        const fileHandle = await this.outputDirectoryHandle.getFileHandle(
            filename,
            { create: true }
        );

        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
    }
}
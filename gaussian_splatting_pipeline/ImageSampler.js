export class ImageSampler {
    constructor(device, outputFolderPath = 'samples') {
        this.device = device;
        this.outputFolderPath = outputFolderPath;
        this.counter = 0;
    }

    nextId() {
        const id = String(this.counter).padStart(6, '0');
        this.counter++;
        return id;
    }

    async save(colorTexture, depthColorTexture) {
        const id = this.nextId();

        await this.saveColor(colorTexture, `color_${id}.png`);
        await this.saveColor(depthColorTexture, `depth_${id}.png`);
    }

    async saveColor(texture, filename) {
        await this.#saveTextureAsPng(
            texture,
            `${this.outputFolderPath}/${filename}`
        );
    }

    async savePair(colorTexture, depthColorTexture) {
        const id = this.nextId();

        await this.saveColor(colorTexture, `color_${id}.png`);
        await this.saveColor(depthColorTexture, `depth_${id}.png`);
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

        await this.#downloadPng(pixels, width, height, filename);
    }

    async #downloadPng(pixels, width, height, filename) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(pixels, width, height);

        ctx.putImageData(imageData, 0, 0);

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/png');
        });

        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename.replaceAll('/', '_');
        a.click();

        URL.revokeObjectURL(url);
    }
}
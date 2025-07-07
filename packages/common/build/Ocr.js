import { Detection, Recognition, YoloDetection } from './models';
export class Ocr {
    #detection;
    #yoloDetection;
    #recognition;
    constructor({ detection, recognition, yoloDetection, }) {
        this.#detection = detection;
        this.#recognition = recognition;
        this.#yoloDetection = yoloDetection;
    }
    static async create(options) {
        const yoloDetection = await YoloDetection.create(options);
        const detection = await Detection.create(options);
        const recognition = await Recognition.create(options);
        return new Ocr({ detection, recognition, yoloDetection });
    }
    async detect(image, options = {}) {
        // 1. run YOLO pipeline
        const yoloImages = await this.runYolo(image, options);
        return this.runOcr(yoloImages, options);
    }
    async runYolo(image, options = {}) {
        return this.#yoloDetection.run(image, options);
    }
    async runOcr(images, options = {}) {
        const texts = [];
        for (const image of images) {
            // run ocr pipeline for each YOLO output
            const lineImages = await this.#detection.run(image, options);
            texts.push(...await this.#recognition.run(lineImages, options));
        }
        return texts;
    }
}
//# sourceMappingURL=Ocr.js.map
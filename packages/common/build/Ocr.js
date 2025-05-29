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
    static async create(options = {}) {
        const yoloDetection = await YoloDetection.create(options);
        const detection = await Detection.create(options);
        const recognition = await Recognition.create(options);
        return new Ocr({ detection, recognition, yoloDetection });
    }
    async detect(image, options = {}) {
        const texts = [];
        const yoloImages = await this.#yoloDetection.run(image, options);
        for (const yoloImage of yoloImages) {
            const lineImages = await this.#detection.run(yoloImage, options);
            texts.push(...await this.#recognition.run(lineImages, options));
        }
        return texts;
    }
}
//# sourceMappingURL=Ocr.js.map
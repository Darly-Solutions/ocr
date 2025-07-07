import type {ModelCreateOptions} from '#common/types'
import {Detection, Recognition, YoloDetection} from './models'

export class Ocr {
    #detection: Detection
    #yoloDetection: YoloDetection
    #recognition: Recognition

    constructor({
                    detection,
                    recognition,
                    yoloDetection,
                }: {
        detection: Detection
        recognition: Recognition
        yoloDetection: YoloDetection
    }) {
        this.#detection = detection
        this.#recognition = recognition
        this.#yoloDetection = yoloDetection
    }

    static async create(options: ModelCreateOptions) {
        const yoloDetection = await YoloDetection.create(options)
        const detection = await Detection.create(options)
        const recognition = await Recognition.create(options)
        return new Ocr({detection, recognition, yoloDetection})
    }

    async detect(image: string, options = {}) {
        // 1. run YOLO pipeline
        const yoloImages = await this.runYolo(image, options);

        return this.runOcr(yoloImages, options);
    }

    async runYolo(image: string, options = {}): Promise<string[]> {
        return this.#yoloDetection.run(image, options);
    }

    async runOcr(images: string[], options = {}) {
        const texts = [];
        for (const image of images) {
            // run ocr pipeline for each YOLO output
            const lineImages = await this.#detection.run(image, options)
            texts.push(...await this.#recognition.run(lineImages, options))
        }

        return texts
    }
}

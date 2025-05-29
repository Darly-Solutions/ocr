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

    static async create(options: ModelCreateOptions = {}) {
        const yoloDetection = await YoloDetection.create(options)
        const detection = await Detection.create(options)
        const recognition = await Recognition.create(options)
        return new Ocr({detection, recognition, yoloDetection})
    }

    async detect(image: string, options = {}) {
        const texts = [];
        const yoloImages = await this.#yoloDetection.run(image, options);
        for (const yoloImage of yoloImages) {
            const lineImages = await this.#detection.run(yoloImage, options)
            texts.push(...await this.#recognition.run(lineImages, options))
        }
        return texts
    }
}

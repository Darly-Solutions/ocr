import type {Line, ModelCreateOptions} from '#common/types'
import {Detection, Recognition} from './models'

export class Ocr {
    #detection: Detection
    #recognition: Recognition

    constructor({detection, recognition}: { detection: Detection, recognition: Recognition }) {
        this.#detection = detection
        this.#recognition = recognition
    }

    static async create(options: ModelCreateOptions) {
        const detection = await Detection.create(options)
        const recognition = await Recognition.create(options)
        return new Ocr({detection, recognition})
    }

    async detect(image: string, options = {}): Promise<Line[]> {
        // run ocr pipeline for each YOLO output
        const lineImages = await this.#detection.run(image, options);

        return await this.#recognition.run(lineImages, options);
    }
}

import { Detection, Recognition } from './models';
export class Ocr {
  #detection;
  #recognition;
  constructor({ detection, recognition }) {
    this.#detection = detection;
    this.#recognition = recognition;
  }
  static async create(options) {
    const detection = await Detection.create(options);
    const recognition = await Recognition.create(options);
    return new Ocr({ detection, recognition });
  }
  async detect(image, options = {}) {
    // run ocr pipeline for each YOLO output
    // const lineImages = await this.#detection.run(image, options);
    // return await this.#recognition.run(lineImages, options);

    const [lineImages, fileName] = await this.#detection.run(image, options);
    return await this.#recognition.run(lineImages, options, fileName);
  }
}
//# sourceMappingURL=Ocr.js.map

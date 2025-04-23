import invariant from 'tiny-invariant';
import { defaultModels, ImageRaw, InferenceSession, splitIntoLineImages } from '../backend';
import { ModelBase } from './ModelBase';
const BASE_SIZE = 32;
export class Detection extends ModelBase {
    static async create({ models, onnxOptions = {}, ...restOptions }) {
        const detectionPath = models?.detectionPath || defaultModels?.detectionPath;
        invariant(detectionPath, 'detectionPath is required');
        const model = await InferenceSession.create(detectionPath, onnxOptions);
        return new Detection({ model, options: restOptions });
    }
    async run(path, { onnxOptions = {} } = {}) {
        const image = await ImageRaw.open(path);
        // Resize image to multiple of 32
        //   - image width and height must be a multiple of 32
        //   - bigger image -> more accurate result, but takes longer time
        // inputImage = await Image.resize(image, multipleOfBaseSize(image, { maxSize: 960 }))
        const inputImage = await image.resize(multipleOfBaseSize(image));
        // this.debugImage(inputImage, 'out1-multiple-of-base-size.jpg')
        // Covert image data to model data
        //   - Using `(RGB / 255 - mean) / std` formula
        //   - omit reshapeOptions (mean/std) is more accurate, can creaet a run option for them
        const modelData = this.imageToInput(inputImage, {
        // mean: [0.485, 0.456, 0.406],
        // std: [0.229, 0.224, 0.225],
        });
        // Run the model
        // console.time('Detection')
        const modelOutput = await this.runModel({ modelData, onnxOptions });
        // console.timeEnd('Detection')
        // Convert output data back to image data
        //   - output value is from 0 to 1, a probability, if value > 0.3, it is a text
        //   - returns a black and white image
        const outputImage = outputToImage(modelOutput, 0.03);
        // this.debugImage(outputImage, 'out2-black-white.jpg')
        // Find text boxes, split image into lines
        //   - findContours from the image
        //   - returns text boxes and line images
        const lineImages = await splitIntoLineImages(outputImage, inputImage);
        this.debugBoxImage(inputImage, lineImages, 'boxes.jpg');
        return this.filterText(lineImages, inputImage.width, inputImage.height);
    }
    filterText(lineImages, frameWidth, frameHeight) {
        return lineImages.filter(lineImage => {
            const box = lineImage.box;
            return isTextAreaValid(box, frameWidth, frameHeight) &&
                !isTextSkewed(box) &&
                !isSubtitleOrHeading(box, frameWidth, frameHeight);
        });
    }
}

/**
 * Перевіряє чи розмір прямокутника з текстом знаходиться в
 * допустимому діапазоні. Дрібний текст у нижній частині кадру
 * дозволяємо незалежно від X‑координати.
 */
function isTextAreaValid(
    textBox,
    frameWidth,
    frameHeight,
    MIN_PCT = 0.0037,
    MAX_PCT = 0.9
  ) {
    const [x1, y1] = textBox[0];
    const [x2, y2] = textBox[2];
    const w = x2 - x1;
    const h = y2 - y1;
    const pct = (w * h) / (frameWidth * frameHeight);
  
    // 1️⃣ дуже маленький текст
    if (pct < MIN_PCT) {
      //     – якщо він у нижній 25 % кадру → залишаємо,
      //       бо це можуть бути субтитри будь‑де по ширині
      if (y1 > 0.75 * frameHeight) return true;
      //     – інакше відсіюємо
      return false;
    }
  
    // 2️⃣ занадто великий блок → відсіюємо
    if (pct > MAX_PCT) return false;
  
    // 3️⃣ нормальний розмір
    return true;
  }

function getTextAngle(box) {
    const [x1, y1, x2, y2] = box;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const angleRad = Math.atan2(dy, dx);
    const angleDeg = angleRad * (180 / Math.PI);
    return Math.abs(angleDeg);
}
function isTextSkewed(box, MAX_SKEW_ANGLE = 2) {
    const angle = getTextAngle(box);
    return angle > MAX_SKEW_ANGLE;
}
function isSubtitleOrHeading(textBox, frameWidth, frameHeight, horizontalTolerance = 0.07, verticalTolerance = 0.15) {
    const [x1, y1, x2, y2] = textBox;
    const textCenterY = (y1 + y2) / 2;
    const textCenterX = (x1 + x2) / 2;
    const centerY = frameHeight / 2;
    const centerX = frameWidth / 2;
    const toleranceY = frameHeight * verticalTolerance;
    const toleranceX = frameWidth * horizontalTolerance;
    if ((textCenterY < frameHeight * 0.2 || textCenterY > frameHeight * 0.8) ||
        (textCenterY > frameHeight * 0.45 || textCenterY < frameHeight * 0.55) &&
            (centerX - toleranceX <= textCenterX && textCenterX <= centerX + toleranceX)) {
        return true;
    }
    return false;
}
function multipleOfBaseSize(image, { maxSize } = {}) {
    let width = image.width;
    let height = image.height;
    if (maxSize && Math.max(width, height) > maxSize) {
        const ratio = width > height ? maxSize / width : maxSize / height;
        width = width * ratio;
        height = height * ratio;
    }
    const newWidth = Math.max(
    // Math.round
    // Math.ceil
    Math.ceil(width / BASE_SIZE) * BASE_SIZE, BASE_SIZE);
    const newHeight = Math.max(Math.ceil(height / BASE_SIZE) * BASE_SIZE, BASE_SIZE);
    return { width: newWidth, height: newHeight };
}
function outputToImage(output, threshold) {
    const height = output.dims[2];
    const width = output.dims[3];
    const data = new Uint8Array(width * height * 4);
    for (const [outIndex, outValue] of output.data.entries()) {
        const n = outIndex * 4;
        const value = outValue > threshold ? 255 : 0;
        data[n] = value; // R
        data[n + 1] = value; // G
        data[n + 2] = value; // B
        data[n + 3] = 255; // A
    }
    return new ImageRaw({ data, width, height });
}
//# sourceMappingURL=Detection.js.map
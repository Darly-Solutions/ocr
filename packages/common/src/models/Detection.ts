import type { Tensor } from 'onnxruntime-common'
import { ImageRaw, InferenceSession, splitIntoLineImages } from '#/backend'
import type { ModelBaseConstructorArgs, Size } from '#/types'
import { ModelBase } from './ModelBase'

const BASE_SIZE = 32

export class Detection extends ModelBase {
  static async create({ modelPath }: { modelPath: string }) {
    const model = await InferenceSession.create(modelPath)
    return new Detection({ model })
  }

  constructor(args: ModelBaseConstructorArgs) {
    super(args)
  }

  async run(path: string, { isDebug } = { isDebug: false }) {
    this.isDebug = isDebug

    const image = await ImageRaw.open(path)

    // Resize image to multiple of 32
    //   - image width and height must be a multiple of 32
    //   - bigger image -> more accurate result, but takes longer time
    // inputImage = await Image.resize(image, multipleOfBaseSize(image, { maxSize: 960 }))
    const inputImage = await image.resize(multipleOfBaseSize(image))
    this.debugImage(inputImage, './output/out1-multiple-of-base-size.jpg')

    // Covert image data to model data
    //   - Using `(RGB / 255 - mean) / std` formula
    //   - omit reshapeOptions (mean/std) is more accurate, can creaet a run option for them
    const modelData = this.imageToInput(inputImage, {
      // mean: [0.485, 0.456, 0.406],
      // std: [0.229, 0.224, 0.225],
    })

    // Run the model
    console.time('Detection')
    const modelOutput = await this.runModel(modelData)
    console.timeEnd('Detection')

    // Convert output data back to image data
    //   - output value is from 0 to 1, a probability, if value > 0.3, it is a text
    //   - returns a black and white image
    const outputImage = outputToImage(modelOutput, 0.03)
    this.debugImage(outputImage, './output/out2-black-white.jpg')

    // Find text boxes, split image into lines
    //   - findContours from the image
    //   - returns text boxes and line images
    const lineImages = await splitIntoLineImages(outputImage, inputImage)
    if (this.isDebug) {
      ;(await inputImage.drawBox(lineImages)).write('./output/out3-boxes.jpg')
    }

    return lineImages
  }
}

function multipleOfBaseSize(image: ImageRaw, { maxSize }: { maxSize?: number } = {}): Size {
  let width = image.width
  let height = image.height
  if (maxSize && Math.max(width, height) > maxSize) {
    const ratio = width > height ? maxSize / width : maxSize / height
    width = width * ratio
    height = height * ratio
  }
  const newWidth = Math.max(
    // Math.round
    // Math.ceil
    Math.ceil(width / BASE_SIZE) * BASE_SIZE,
    BASE_SIZE,
  )
  const newHeight = Math.max(Math.ceil(height / BASE_SIZE) * BASE_SIZE, BASE_SIZE)
  return { width: newWidth, height: newHeight }
}

function outputToImage(output: Tensor, threshold: number): ImageRaw {
  const height = output.dims[2]
  const width = output.dims[3]
  const data = new Uint8Array(width * height * 4)
  for (const [outIndex, outValue] of output.data.entries()) {
    const n = outIndex * 4
    const value = (outValue as number) > threshold ? 255 : 0
    data[n] = value // R
    data[n + 1] = value // G
    data[n + 2] = value // B
    data[n + 3] = 255 // A
  }
  return new ImageRaw({ data, width, height })
}

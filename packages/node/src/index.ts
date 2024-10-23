import fs from 'node:fs/promises'
import BaseOcr, { registerBackend, type ModelCreateOptions } from '@darly-solutions/ocr-common'
import { splitIntoLineImages } from '@darly-solutions/ocr-common/splitIntoLineImages'
import defaultModels from '@darly-solutions/ocr-models/node'
import { InferenceSession } from 'onnxruntime-node'
import { FileUtils } from './FileUtils'
import { ImageRaw } from './ImageRaw'

registerBackend({
  FileUtils,
  ImageRaw,
  InferenceSession,
  splitIntoLineImages,
  defaultModels,
})

// biome-ignore lint/complexity/noStaticOnlyClass: <explanation>
class Ocr extends BaseOcr {
  static async create(options: ModelCreateOptions = {}) {
    const ocr = await BaseOcr.create(options)
    if (options.debugOutputDir) {
      await fs.mkdir(options.debugOutputDir, { recursive: true })
    }
    return ocr
  }
}

export * from '@darly-solutions/ocr-common'

export default Ocr

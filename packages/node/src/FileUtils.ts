import { promises as fs } from 'fs';
import { FileUtilsBase } from '@darly-solutions/ocr-common'

export class FileUtils extends FileUtilsBase {
  static async read(path: string) {
    return await fs.readFile(path, 'utf8')
  }
}

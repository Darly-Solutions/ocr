import { promises as fs } from 'fs';
import { FileUtilsBase } from '@darly-solutions/ocr-common';
export class FileUtils extends FileUtilsBase {
    static async read(path) {
        return await fs.readFile(path, 'utf8');
    }
}
//# sourceMappingURL=FileUtils.js.map
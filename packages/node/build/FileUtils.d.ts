import { FileUtilsBase } from '@darly-solutions/ocr-common';
export declare class FileUtils extends FileUtilsBase {
    static read(path: string): Promise<string>;
}

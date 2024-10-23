import BaseOcr, { type ModelCreateOptions } from '@darly-solutions/ocr-common';
declare class Ocr extends BaseOcr {
    static create(options?: ModelCreateOptions): Promise<BaseOcr>;
}
export * from '@darly-solutions/ocr-common';
export default Ocr;

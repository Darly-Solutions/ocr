import type { ImageRawData } from '../types';
export declare class ImageRawBase {
    data: ImageRawData['data'];
    width: ImageRawData['width'];
    height: ImageRawData['height'];
    constructor({ data, width, height }: ImageRawData);
    getImageRawData(): ImageRawData;
}

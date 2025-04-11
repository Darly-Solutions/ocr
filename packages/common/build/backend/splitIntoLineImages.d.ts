import type { LineImage, ImageRaw as ImageRawType } from '../types';
export declare function splitIntoLineImages(image: ImageRawType, sourceImage: ImageRawType): Promise<LineImage[]>;

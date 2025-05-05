import type { ImageRaw as ImageRawType, LineImage } from '../types';
export declare function splitIntoLineImages(image: ImageRawType, sourceImage: ImageRawType): Promise<LineImage[]>;

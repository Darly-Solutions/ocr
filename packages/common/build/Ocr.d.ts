import type { ModelCreateOptions } from './types';
import { Detection, Recognition, YoloDetection } from './models';
export declare class Ocr {
    #private;
    constructor({ detection, recognition, yoloDetection, }: {
        detection: Detection;
        recognition: Recognition;
        yoloDetection: YoloDetection;
    });
    static create(options?: ModelCreateOptions): Promise<Ocr>;
    detect(image: string, options?: {}): Promise<import("./types").Line[]>;
}

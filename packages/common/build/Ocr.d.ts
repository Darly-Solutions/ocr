import type { ModelCreateOptions } from './types';
import { Detection, Recognition } from './models';
export declare class Ocr {
    #private;
    static create(options?: ModelCreateOptions): Promise<Ocr>;
    constructor({ detection, recognition, }: {
        detection: Detection;
        recognition: Recognition;
    });
    detect(image: string, options?: {}): Promise<import("./types").Line[]>;
}

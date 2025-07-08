import type { Line, ModelCreateOptions } from './types';
import { Detection, Recognition } from './models';
export declare class Ocr {
    #private;
    constructor({ detection, recognition }: {
        detection: Detection;
        recognition: Recognition;
    });
    static create(options: ModelCreateOptions): Promise<Ocr>;
    detect(image: string, options?: {}): Promise<Line[]>;
}

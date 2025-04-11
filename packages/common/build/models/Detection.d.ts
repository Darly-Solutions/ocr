import type { InferenceSession as InferenceSessionCommon } from 'onnxruntime-common';
import type { LineImage, ModelCreateOptions } from '../types';
import { ModelBase } from './ModelBase';
export declare class Detection extends ModelBase {
    static create({ models, onnxOptions, ...restOptions }: ModelCreateOptions): Promise<Detection>;
    run(path: string, { onnxOptions }?: {
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<LineImage[]>;
    filterText(lineImages: LineImage[], frameWidth: number, frameHeight: number): LineImage[];
}

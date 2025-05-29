import type { LineImage, ModelCreateOptions } from '../types';
import type { InferenceSession as InferenceSessionCommon } from 'onnxruntime-common';
import { ModelBase } from './ModelBase';
export declare class Detection extends ModelBase {
    static create({ models, onnxOptions, ...restOptions }: ModelCreateOptions): Promise<Detection>;
    run(image: any, { onnxOptions }?: {
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<LineImage[]>;
}

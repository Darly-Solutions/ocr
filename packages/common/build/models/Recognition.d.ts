import type { Dictionary, Line, LineImage, ModelBaseConstructorArg, ModelCreateOptions } from '../types';
import type { InferenceSession as InferenceSessionCommon, Tensor } from 'onnxruntime-common';
import { ModelBase } from './ModelBase';
export declare class Recognition extends ModelBase {
    #private;
    constructor(options: ModelBaseConstructorArg, dictionary: Dictionary);
    static create({ models, onnxOptions, ...restOptions }: ModelCreateOptions): Promise<Recognition>;
    run(lineImages: LineImage[], { onnxOptions }?: {
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<Line[]>;
    decodeText(output: Tensor): Line[];
}

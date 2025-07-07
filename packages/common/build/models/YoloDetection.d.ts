import type { ImageRaw as ImageRawType, ModelCreateOptions, ModelData } from '../types';
import type { InferenceSession as InferenceSessionCommon } from 'onnxruntime-common';
import { ModelBase } from './ModelBase';
type Box = [number, number, number, number, string, number];
export declare class YoloDetection extends ModelBase {
    static create({ models, onnxOptions, ...restOptions }: ModelCreateOptions): Promise<YoloDetection>;
    run(imagePath: string, { onnxOptions }?: {
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<string[]>;
    imageToInput(image: ImageRawType): {
        data: number[];
        width: number;
        height: number;
    };
    prepareInput(modelData: ModelData): import("onnxruntime-common").TypedTensor<"float32">;
    processOutput(output: any): Box[];
    saveImage(image: ImageRawType | any, path: string): Promise<string>;
}
export declare function splitIntoLineImagesNew(imageElement: ImageRawType, box: Box): any;
export {};

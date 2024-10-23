import { InferenceSession } from 'onnxruntime-common';
import { ImageRawBase as ImageRaw } from '../backend/ImageRawBase.js';
import type { splitIntoLineImages } from '../backend/splitIntoLineImages.js';
export { FileUtilsBase as FileUtils } from '../backend/FileUtilsBase.js';
export { ImageRaw, InferenceSession };
export type SplitIntoLineImages = typeof splitIntoLineImages;
export type ReshapeOptions = {
    mean?: number[];
    std?: number[];
};
export type ImageRawData = {
    data: Uint8Array | Uint8ClampedArray | Buffer;
    width: number;
    height: number;
};
export type ModelData = {
    data: number[] | Uint8Array;
    width: number;
    height: number;
};
export type Size = {
    width: number;
    height: number;
};
export type SizeOption = {
    width?: number;
    height?: number;
    fit?: 'contain' | 'cover' | 'fill' | 'inside' | 'outside';
};
export type LineImage = {
    image: ImageRaw;
    box: number[][];
};
export type Region = {
    left: number;
    top: number;
    width: number;
    height: number;
};
export type Line = {
    text: string;
    mean: number;
    box?: number[][];
};
export type Dictionary = string[];
export type Point = [x: number, y: number];
export interface ModelBaseConstructorArg {
    model: InferenceSession;
    options: ModelBaseOptions;
}
export interface ModelBaseOptions {
    isDebug?: boolean;
    debugOutputDir?: string;
    accuracyMean?: number;
}
export interface ModelCreateOptions extends ModelBaseOptions {
    models?: {
        detectionPath: string;
        recognitionPath: string;
        dictionaryPath: string;
    };
    onnxOptions?: InferenceSession.SessionOptions;
}

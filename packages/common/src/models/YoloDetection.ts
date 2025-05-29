import {ImageRaw, InferenceSession} from '#common/backend'
import type {ImageRaw as ImageRawType, ModelCreateOptions, ModelData} from '#common/types'
import {Mat} from "@techstark/opencv-js";
import type {InferenceSession as InferenceSessionCommon} from 'onnxruntime-common'
import {Tensor} from 'onnxruntime-common'
import invariant from 'tiny-invariant'
import {ModelBase} from './ModelBase'


type Box = [number, number, number, number, string, number];

const yolo_classes = [
    'EDITORIAL'
];

export class YoloDetection extends ModelBase {
    static async create({models, onnxOptions = {}, ...restOptions}: ModelCreateOptions) {
        const detectionPath = models?.yoloDetectionPath
        invariant(detectionPath, 'yoloDetectionPath is required')
        const model = await InferenceSession.create(detectionPath, onnxOptions)
        return new YoloDetection({model, options: restOptions})
    }

    async run(path: string, {onnxOptions = {}}: { onnxOptions?: InferenceSessionCommon.RunOptions } = {}) {
        const image = await ImageRaw.open(path);

        const inputImage = await image.resize({width: 640, height: 640, fit: 'fill'});

        const modelData = this.imageToInput(inputImage);

        const modelOutput = await this.runModel({modelData, onnxOptions});
        const processed = this.processOutput(modelOutput.data);

        const croppedImages = [];

        for (const item of processed) {
            croppedImages.push(...splitIntoLineImagesNew(inputImage, item));
        }

        croppedImages.forEach((lineImage, index) => {
            this.debugImage(lineImage.image, `yolo_box_${Math.random()}_${index}.jpg`)
        });


        console.log('croppedImages', croppedImages);
        return croppedImages;
    }

    imageToInput(image: ImageRawType) {
        const R = [];
        const G = [];
        const B = [];
        for (let i = 0; i < image.data.length; i += 4) {
            R.push(image.data[i] / 255.0);
            G.push(image.data[i + 1] / 255.0);
            B.push(image.data[i + 2] / 255.0);
        }
        const newData = [...R, ...G, ...B];
        return {
            data: newData,
            width: image.width,
            height: image.height,
        };
    }


    prepareInput(modelData: ModelData) {
        const input = Float32Array.from(modelData.data);
        return new Tensor(input, [1, 3, modelData.height, modelData.width]);
    }

    processOutput(output: any): Box[] {
        let boxes: Box[] = [];
        for (let index = 0; index < 8400; index++) {
            const [class_id, prob] = [...Array(80).keys()]
                .map(col => [col, output[8400 * (col + 4) + index]] as [number, number])
                .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0] as [number, number]);
            if (prob < 0.5) {
                continue;
            }
            const label = yolo_classes[class_id];
            const xc = output[index];
            const yc = output[8400 + index];
            const w = output[2 * 8400 + index];
            const h = output[3 * 8400 + index];
            const x1 = (xc - w / 2);
            const y1 = (yc - h / 2);
            const x2 = (xc + w / 2);
            const y2 = (yc + h / 2);
            boxes.push([x1, y1, x2, y2, label, prob]);
        }

        boxes = boxes.sort((box1, box2) => box2[5] - box1[5])
        const result: Box[] = [];
        while (boxes.length > 0) {
            result.push(boxes[0]);
            boxes = boxes.filter(box => iou(boxes[0], box) < 0.7);
        }

        return result;
    }
}


/**
 * Function calculates "Intersection-over-union" coefficient for specified two boxes
 * https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
 * @param box1 First box in format: [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format: [x1,y1,x2,y2,object_class,probability]
 * @returns Intersection over union ratio as a float number
 */
function iou(box1: Box, box2: Box): number {
    return intersection(box1, box2) / union(box1, box2);
}

/**
 * Function calculates union area of two boxes.
 *     :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
 *     :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
 *     :return: Area of the boxes union as a float number
 * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
 * @returns Area of the boxes union as a float number
 */
function union(box1: Box, box2: Box): number {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)
}

/**
 * Function calculates intersection area of two boxes
 * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
 * @returns Area of intersection of the boxes as a float number
 */
function intersection(box1: Box, box2: Box): number {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const x1 = Math.max(box1_x1, box2_x1);
    const y1 = Math.max(box1_y1, box2_y1);
    const x2 = Math.min(box1_x2, box2_x2);
    const y2 = Math.min(box1_y2, box2_y2);
    return (x2 - x1) * (y2 - y1)
}

export function splitIntoLineImagesNew(imageElement: ImageRawType, box: Box) {
    const [x1, y1, x2, y2] = box;

    const src = cvImread(imageElement);

    let width = x2 - x1;
    let height = y2 - y1;

    let extraWidth = width * 0.15;
    let extraHeight = height * 0.15;

    let newWidth = width + extraWidth;
    let newHeight = height + extraHeight;

    let centerX = x1 + width / 2;
    let centerY = y1 + height / 2;

    let newX1 = Math.max(0, Math.floor(centerX - newWidth / 2));
    let newY1 = Math.max(0, Math.floor(centerY - newHeight / 2));
    newWidth = Math.min(newWidth, src.cols - newX1);
    newHeight = Math.min(newHeight, src.rows - newY1);

    let rect = new cv.Rect(newX1, newY1, Math.floor(newWidth), Math.floor(newHeight));
    let roiMat = src.roi(rect);
    let cropped = new cv.Mat();
    roiMat.copyTo(cropped);

    roiMat.delete();
    src.delete();

    return cvImshow(cropped);
}


function cvImread(image: ImageRawType) {
    return cv.matFromImageData(image);
}

function cvImshow(mat: Mat) {
    return new ImageRaw({data: mat.data, width: mat.cols, height: mat.rows});
}

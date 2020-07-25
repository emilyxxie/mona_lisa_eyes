import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import { Box } from './box';
export interface NormalizedFace {
    topLeft: [number, number] | tf.Tensor1D;
    bottomRight: [number, number] | tf.Tensor1D;
    landmarks?: number[][] | tf.Tensor2D;
    probability?: number | tf.Tensor1D;
}
export declare type BlazeFacePrediction = {
    box: Box;
    landmarks: tf.Tensor2D;
    probability: tf.Tensor1D;
    anchor: tf.Tensor2D | [number, number];
};
export declare class BlazeFaceModel {
    private blazeFaceModel;
    private width;
    private height;
    private maxFaces;
    private anchors;
    private anchorsData;
    private inputSize;
    private inputSizeData;
    private iouThreshold;
    private scoreThreshold;
    constructor(model: tfconv.GraphModel, width: number, height: number, maxFaces: number, iouThreshold: number, scoreThreshold: number);
    getBoundingBoxes(inputImage: tf.Tensor4D, returnTensors: boolean, annotateBoxes?: boolean): Promise<{
        boxes: Array<BlazeFacePrediction | Box>;
        scaleFactor: tf.Tensor | [number, number];
    }>;
    estimateFaces(input: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement | HTMLCanvasElement, returnTensors?: boolean, flipHorizontal?: boolean, annotateBoxes?: boolean): Promise<NormalizedFace[]>;
}

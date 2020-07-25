import { BlazeFaceModel } from './face';
export declare function load({ maxFaces, inputWidth, inputHeight, iouThreshold, scoreThreshold }?: {
    maxFaces?: number;
    inputWidth?: number;
    inputHeight?: number;
    iouThreshold?: number;
    scoreThreshold?: number;
}): Promise<BlazeFaceModel>;
export { NormalizedFace, BlazeFaceModel, BlazeFacePrediction } from './face';

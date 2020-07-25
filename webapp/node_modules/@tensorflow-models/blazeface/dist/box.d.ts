import * as tf from '@tensorflow/tfjs-core';
export declare type Box = {
    startEndTensor: tf.Tensor2D;
    startPoint: tf.Tensor2D;
    endPoint: tf.Tensor2D;
};
export declare const disposeBox: (box: Box) => void;
export declare const createBox: (startEndTensor: tf.Tensor<tf.Rank.R2>) => Box;
export declare const scaleBox: (box: Box, factors: tf.Tensor<tf.Rank.R1> | [number, number]) => Box;

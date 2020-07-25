/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { Tensor } from '../tensor';
import { TensorLike } from '../types';
export declare enum Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2,
    SUM_BY_NONZERO_WEIGHTS = 3
}
/**
 * Computes the weighted loss between two tensors.
 *
 * @param losses Tensor of shape `[batch_size, d1, ... dN]`.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `losses`, and must be broadcastable to `losses` (i.e., all
 *    dimensions must be either `1`, or the same as the corresponding
 *    `losses` dimension).
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function computeWeightedLoss_<T extends Tensor, O extends Tensor>(losses: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
/**
 * Computes the absolute difference loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function absoluteDifference_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
/**
 * Computes the mean squared error between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function meanSquaredError_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
/**
 * Computes the cosine distance loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param axis The dimension along which the cosine distance is computed.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function cosineDistance_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, axis: number, weights?: Tensor | TensorLike, reduction?: Reduction): O;
/**
 * Computes the Hinge loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function hingeLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
/**
 * Computes the log loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param epsilon A small increment to avoid taking log of zero
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function logLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, epsilon?: number, reduction?: Reduction): O;
/**
 * Computes the sigmoid cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newMulticlassLabels = multiclassLabels * (1 - labelSmoothing)
 *                         + 0.5 * labelSmoothing
 *
 * @param multiClassLabels The ground truth output tensor of shape
 * [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' } */
declare function sigmoidCrossEntropy_<T extends Tensor, O extends Tensor>(multiClassLabels: T | TensorLike, logits: T | TensorLike, weights?: Tensor | TensorLike, labelSmoothing?: number, reduction?: Reduction): O;
/**
 * Computes the huber loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param delta Point where huber loss changes from quadratic to linear.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`.
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
declare function huberLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, delta?: number, reduction?: Reduction): O;
/**
 * Computes the softmax cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newOnehotLabels = onehotLabels * (1 - labelSmoothing)
 *                         + labelSmoothing / numClasses
 *
 * @param onehotLabels One hot encoded labels
 *    [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or 1, and must be
 *    broadcastable to `loss`  of shape [batch_size]
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' } */
declare function softmaxCrossEntropy_<T extends Tensor, O extends Tensor>(onehotLabels: T | TensorLike, logits: T | TensorLike, weights?: Tensor | TensorLike, labelSmoothing?: number, reduction?: Reduction): O;
export declare const absoluteDifference: typeof absoluteDifference_;
export declare const computeWeightedLoss: typeof computeWeightedLoss_;
export declare const cosineDistance: typeof cosineDistance_;
export declare const hingeLoss: typeof hingeLoss_;
export declare const huberLoss: typeof huberLoss_;
export declare const logLoss: typeof logLoss_;
export declare const meanSquaredError: typeof meanSquaredError_;
export declare const sigmoidCrossEntropy: typeof sigmoidCrossEntropy_;
export declare const softmaxCrossEntropy: typeof softmaxCrossEntropy_;
export {};

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import { ExplicitPadding } from '../src/ops/conv_util';
import { NamedTensorInfoMap, TensorInfo } from './kernel_registry';
import { DataType, PixelData } from './types';
export declare const Add = "Add";
export declare type AddInputs = BinaryInputs;
export declare const AddN = "AddN";
export declare type AddNInputs = TensorInfo[];
export declare const Atan2 = "Atan2";
export declare type Atan2Inputs = BinaryInputs;
export declare const AvgPool = "AvgPool";
export declare type AvgPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface AvgPoolAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const AvgPoolBackprop = "AvgPoolBackprop";
export declare type AvgPoolBackpropInputs = Pick<NamedTensorInfoMap, 'dy' | 'input'>;
export interface AvgPoolBackpropAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
}
export declare const AvgPool3D = "AvgPool3D";
export declare type AvgPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface AvgPool3DAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
    dataFormat: 'NDHWC' | 'NCDHW';
    dilations?: [number, number, number] | number;
}
export declare const AvgPool3DBackprop = "AvgPool3DBackprop";
export declare type AvgPool3DBackpropInputs = Pick<NamedTensorInfoMap, 'dy' | 'input'>;
export interface AvgPool3DBackpropAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dilations: [number, number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const BatchMatMul = "BatchMatMul";
export declare type BatchMatMulInputs = Pick<NamedTensorInfoMap, 'a' | 'b'>;
export interface BatchMatMulAttrs {
    transposeA: boolean;
    transposeB: boolean;
}
export declare const BatchToSpaceND = "BatchToSpaceND";
export declare type BatchToSpaceNDInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface BatchToSpaceNDAttrs {
    blockShape: number[];
    crops: number[][];
}
export declare type BinaryInputs = Pick<NamedTensorInfoMap, 'a' | 'b'>;
export declare const BroadcastTo = "BroadcastTo";
export declare type BroadcastToInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface BroadCastToAttrs {
    shape: number[];
    inputShape: number[];
}
export declare const Complex = "Complex";
export declare type ComplexInputs = Pick<NamedTensorInfoMap, 'real' | 'imag'>;
export declare const Concat = "Concat";
export declare type ConcatInputs = TensorInfo[];
export interface ConcatAttrs {
    axis: number;
}
export declare const Conv2D = "Conv2D";
export declare type Conv2DInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
export interface Conv2DAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dilations: [number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const Conv2DBackpropFilter = "Conv2DBackpropFilter";
export declare type Conv2DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x' | 'dy'>;
export interface Conv2DBackpropFilterAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const Conv2DBackpropInput = "Conv2DBackpropInput";
export declare type Conv2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy' | 'filter'>;
export interface Conv2DBackpropInputAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const Conv3D = "Conv3D";
export declare type Conv3DInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
export interface Conv3DAttrs {
    strides: [number, number, number] | number;
    pad: 'valid' | 'same';
    dataFormat: 'NDHWC' | 'NCDHW';
    dilations: [number, number, number] | number;
}
export declare const Conv3DBackpropFilterV2 = "Conv3DBackpropFilterV2";
export declare type Conv3DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x' | 'y'>;
export interface Conv3DBackpropFilterAttrs {
    strides: [number, number, number] | number;
    pad: 'valid' | 'same';
}
export declare const Conv3DBackpropInputV2 = "Conv3DBackpropInputV2";
export declare type Conv3DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'>;
export interface Conv3DBackpropInputAttrs {
    pad: 'valid' | 'same';
}
export declare const Cumsum = "Cumsum";
export declare type CumsumInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface CumsumAttrs {
    axis: number;
    exclusive: boolean;
    reverse: boolean;
}
export declare const DepthToSpace = "DepthToSpace";
export declare type DepthToSpaceInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface DepthToSpaceAttrs {
    blockSize: number;
    dataFormat: 'NHWC' | 'NCHW';
}
export declare const DepthwiseConv2dNative = "DepthwiseConv2dNative";
export declare type DepthwiseConv2dNativeInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
export interface DepthwiseConv2dNativeAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    dataFormat: 'NHWC' | 'NCHW';
    dilations: [number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const DepthwiseConv2dNativeBackpropFilter = "DepthwiseConv2dNativeBackpropFilter";
export declare type DepthwiseConv2dNativeBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x' | 'dy'>;
export declare const DepthwiseConv2dNativeBackpropInput = "DepthwiseConv2dNativeBackpropInput";
export declare type DepthwiseConv2dNativeBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy'>;
export declare const Diag = "Diag";
export declare type DiagInputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const Div = "Div";
export declare type DivInputs = BinaryInputs;
export declare const Elu = "Elu";
export declare type EluInputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const EluGrad = "EluGrad";
export declare type EluGradInputs = Pick<NamedTensorInfoMap, 'dy' | 'y'>;
export declare const Equal = "Equal";
export declare type EqualInputs = BinaryInputs;
export declare const FloorDiv = "FloorDiv";
export declare type FloorDivInputs = BinaryInputs;
export declare const Fill = "Fill";
export interface FillAttrs {
    shape: number[];
    value: number | string;
    dtype: DataType;
}
export declare const FusedBatchNorm = "FusedBatchNorm";
export declare type FusedBatchNormInputs = Pick<NamedTensorInfoMap, 'x' | 'scale' | 'offset' | 'mean' | 'variance'>;
export interface FusedBatchNormAttrs {
    varianceEpsilon: number;
}
export declare const GatherNd = "GatherNd";
export declare type GatherNdInputs = Pick<NamedTensorInfoMap, 'params' | 'indices'>;
export declare const Greater = "Greater";
export declare type GreaterInputs = BinaryInputs;
export declare const GreaterEqual = "GreaterEqual";
export declare type GreaterEqualInputs = BinaryInputs;
export declare const Identity = "Identity";
export declare type IdentityInputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const Imag = "Imag";
export declare type ImagInputs = Pick<NamedTensorInfoMap, 'input'>;
export declare const Less = "Less";
export declare type LessInputs = BinaryInputs;
export declare const LessEqual = "LessEqual";
export declare type LessEqualInputs = BinaryInputs;
export declare const LRN = "LRN";
export declare type LRNInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface LRNAttrs {
    depthRadius: number;
    bias: number;
    alpha: number;
    beta: number;
}
export declare const LRNBackprop = "LRNBackprop";
export declare type LRNBackpropInputs = Pick<NamedTensorInfoMap, 'x' | 'y' | 'dy'>;
export interface LRNBackpropAttrs {
    depthRadius: number;
    bias: number;
    alpha: number;
    beta: number;
}
export declare const Max = "Max";
export declare type MaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxAttrs {
    reductionIndices: number | number[];
    keepDims: boolean;
}
export declare const Maximum = "Maximum";
export declare type MaximumInputs = BinaryInputs;
export declare const MaxPool = "MaxPool";
export declare type MaxPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPoolAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const MaxPoolBackprop = "MaxPoolBackprop";
export declare type MaxPoolBackpropInputs = Pick<NamedTensorInfoMap, 'dy' | 'input' | 'output'>;
export interface MaxPoolBackpropAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const MaxPool3D = "MaxPool3D";
export declare type MaxPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPool3DAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dataFormat: 'NDHWC' | 'NCDHW';
    dilations?: [number, number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const MaxPool3DBackprop = "MaxPool3DBackprop";
export declare type MaxPool3DBackpropInputs = Pick<NamedTensorInfoMap, 'dy' | 'input' | 'output'>;
export interface MaxPool3DBackpropAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dilations?: [number, number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
export declare const MaxPoolWithArgmax = "MaxPoolWithArgmax";
export declare type MaxPoolWithArgmaxInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface MaxPoolWithArgmaxAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    includeBatchInIndex: boolean;
}
export declare const Minimum = "Minimum";
export declare type MinimumInputs = BinaryInputs;
export declare const Mod = "Mod";
export declare type ModInputs = BinaryInputs;
export declare const Multiply = "Multiply";
export declare type MultiplyInputs = BinaryInputs;
export declare const NotEqual = "NotEqual";
export declare type NotEqualInputs = BinaryInputs;
export declare const NonMaxSuppressionV3 = "NonMaxSuppressionV3";
export declare type NonMaxSuppressionV3Inputs = Pick<NamedTensorInfoMap, 'boxes' | 'scores'>;
export interface NonMaxSuppressionV3Attrs {
    maxOutputSize: number;
    iouThreshold: number;
    scoreThreshold: number;
}
export declare const NonMaxSuppressionV5 = "NonMaxSuppressionV5";
export declare type NonMaxSuppressionV5Inputs = Pick<NamedTensorInfoMap, 'boxes' | 'scores'>;
export interface NonMaxSuppressionV5Attrs {
    maxOutputSize: number;
    iouThreshold: number;
    scoreThreshold: number;
    softNmsSigma: number;
}
export declare const OneHot = "OneHot";
export declare type OneHotInputs = Pick<NamedTensorInfoMap, 'indices'>;
export interface OneHotAttrs {
    depth: number;
    onValue: number;
    offValue: number;
}
export declare const PadV2 = "PadV2";
export declare type PadV2Inputs = Pick<NamedTensorInfoMap, 'x'>;
export interface PadV2Attrs {
    paddings: Array<[number, number]>;
    constantValue: number;
}
export declare const Pool = "Pool";
export declare type PoolInputs = Pick<NamedTensorInfoMap, 'input'>;
export declare const Pow = "Pow";
export declare type PowInputs = BinaryInputs;
export declare const Prelu = "Prelu";
export declare type PreluInputs = Pick<NamedTensorInfoMap, 'x' | 'alpha'>;
export declare const Real = "Real";
export declare type RealInputs = Pick<NamedTensorInfoMap, 'input'>;
export declare const Relu = "Relu";
export declare type ReluInputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const Relu6 = "Relu6";
export declare type Relu6Inputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const SelectV2 = "SelectV2";
export declare type SelectV2Inputs = Pick<NamedTensorInfoMap, 'condition' | 't' | 'e'>;
export declare const Selu = "Selu";
export declare type SeluInputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const SpaceToBatchND = "SpaceToBatchND";
export declare type SpaceToBatchNDInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface SpaceToBatchNDAttrs {
    blockShape: number[];
    paddings: number[][];
}
export declare const SplitV = "SplitV";
export declare type SplitVInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface SplitVAttrs {
    numOrSizeSplits: number[] | number;
    axis: number;
}
export declare const SquaredDifference = "SquaredDifference";
export declare type SquaredDifferenceInputs = BinaryInputs;
export declare const Square = "Square";
export declare type SquareInputs = Pick<NamedTensorInfoMap, 'x'>;
export declare const Sub = "Sub";
export declare type SubInputs = BinaryInputs;
export declare const Tile = "Tile";
export declare type TileInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TileAttrs {
    reps: number[];
}
export declare const Transpose = "Transpose";
export declare type TransposeInputs = Pick<NamedTensorInfoMap, 'x'>;
export interface TransposeAttrs {
    perm: number[];
}
/**
 * TensorFlow.js-only kernels
 */
export declare const FromPixels = "FromPixels";
export interface FromPixelsInputs {
    pixels: PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;
}
export interface FromPixelsAttrs {
    numChannels: number;
}

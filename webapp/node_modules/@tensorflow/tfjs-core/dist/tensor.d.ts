/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import { ArrayMap, BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, Rank, ShapeMap, SingleValueMap, TensorLike, TensorLike1D } from './types';
export interface TensorData<D extends DataType> {
    dataId?: DataId;
    values?: DataTypeMap[D];
}
export interface Backend {
}
/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export declare class TensorBuffer<R extends Rank, D extends DataType = 'float32'> {
    dtype: D;
    size: number;
    shape: ShapeMap[R];
    strides: number[];
    values: DataTypeMap[D];
    constructor(shape: ShapeMap[R], dtype: D, values?: DataTypeMap[D]);
    /**
     * Sets a value in the buffer at a given location.
     *
     * @param value The value to set.
     * @param locs  The location indices.
     */
    /** @doc {heading: 'Tensors', subheading: 'Creation'} */
    set(value: SingleValueMap[D], ...locs: number[]): void;
    /**
     * Returns the value in the buffer at the provided location.
     *
     * @param locs The location indices.
     */
    /** @doc {heading: 'Tensors', subheading: 'Creation'} */
    get(...locs: number[]): SingleValueMap[D];
    locToIndex(locs: number[]): number;
    indexToLoc(index: number): number[];
    readonly rank: number;
    /**
     * Creates an immutable `tf.Tensor` object from the buffer.
     */
    /** @doc {heading: 'Tensors', subheading: 'Creation'} */
    toTensor(): Tensor<R>;
}
export interface TensorTracker {
    makeTensor(values: DataValues, shape: number[], dtype: DataType, backend?: Backend): Tensor;
    makeVariable(initialValue: Tensor, trainable?: boolean, name?: string, dtype?: DataType): Variable;
    incRef(a: Tensor, backend: Backend): void;
    disposeTensor(t: Tensor): void;
    disposeVariable(v: Variable): void;
    read(dataId: DataId): Promise<BackendValues>;
    readSync(dataId: DataId): BackendValues;
}
/**
 * The Tensor class calls into this handler to delegate chaining operations.
 */
export interface OpHandler {
    cast<T extends Tensor>(x: T, dtype: DataType): T;
    buffer<R extends Rank, D extends DataType>(shape: ShapeMap[R], dtype: D, values?: DataTypeMap[D]): TensorBuffer<R, D>;
    print<T extends Tensor>(x: T, verbose: boolean): void;
    reshape<R2 extends Rank>(x: Tensor, shape: ShapeMap[R2]): Tensor<R2>;
    expandDims<R2 extends Rank>(x: Tensor, axis: number): Tensor<R2>;
    squeeze<T extends Tensor>(x: Tensor, axis?: number[]): T;
    clone<T extends Tensor>(x: T): T;
    gather<T extends Tensor>(x: T, indices: Tensor | TensorLike, axis: number): T;
    norm(x: Tensor, ord: number | 'euclidean' | 'fro', axis: number | number[], keepDims: boolean): Tensor;
    slice<R extends Rank, T extends Tensor<R>>(x: T, begin: number | number[], size?: number | number[]): T;
    reverse<T extends Tensor>(x: T, axis?: number | number[]): T;
    stack<T extends Tensor>(tensors: Array<T | TensorLike>, axis: number): Tensor;
    unstack<T extends Tensor>(value: T, axis: number): Tensor[];
    all<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    any<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    logSumExp<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    sum<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    prod<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    mean<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    min<T extends Tensor>(x: Tensor, axis: number | number[], keepDims: boolean): T;
    argMin<T extends Tensor>(x: Tensor, axis: number): T;
    argMax<T extends Tensor>(x: Tensor, axis: number): T;
    addStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    subStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    powStrict<T extends Tensor>(base: T, exp: Tensor | TensorLike): T;
    mulStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    divStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    modStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    minimumStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    maximumStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    squaredDifferenceStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    logicalNot<T extends Tensor>(x: T): T;
    logicalAnd<T extends Tensor>(a: Tensor, b: Tensor | TensorLike): T;
    logicalOr<T extends Tensor>(a: Tensor, b: Tensor | TensorLike): T;
    logicalXor<T extends Tensor>(a: Tensor, b: Tensor | TensorLike): T;
    where<T extends Tensor>(condition: Tensor | TensorLike, a: T, b: T | TensorLike): T;
    notEqualStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    lessStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    equalStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    lessEqualStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    greaterStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    greaterEqualStrict<T extends Tensor>(a: T, b: T | TensorLike): T;
    neg<T extends Tensor>(x: T): T;
    ceil<T extends Tensor>(x: T): T;
    floor<T extends Tensor>(x: T): T;
    sign<T extends Tensor>(x: T): T;
    isNaN<T extends Tensor>(x: T): T;
    isInf<T extends Tensor>(x: T): T;
    isFinite<T extends Tensor>(x: T): T;
    round<T extends Tensor>(x: T): T;
    exp<T extends Tensor>(x: T): T;
    expm1<T extends Tensor>(x: T): T;
    log<T extends Tensor>(x: T): T;
    log1p<T extends Tensor>(x: T): T;
    sqrt<T extends Tensor>(x: T): T;
    rsqrt<T extends Tensor>(x: T): T;
    square<T extends Tensor>(x: T): T;
    reciprocal<T extends Tensor>(x: T): T;
    abs<T extends Tensor>(x: T): T;
    clipByValue<T extends Tensor>(x: T, clipValueMin: number, clipValueMax: number): T;
    sigmoid<T extends Tensor>(x: T): T;
    logSigmoid<T extends Tensor>(x: T): T;
    softplus<T extends Tensor>(x: T): T;
    zerosLike<T extends Tensor>(x: T): T;
    onesLike<T extends Tensor>(x: T): T;
    sin<T extends Tensor>(x: T): T;
    cos<T extends Tensor>(x: T): T;
    tan<T extends Tensor>(x: T): T;
    asin<T extends Tensor>(x: T): T;
    acos<T extends Tensor>(x: T): T;
    atan<T extends Tensor>(x: T): T;
    sinh<T extends Tensor>(x: T): T;
    cosh<T extends Tensor>(x: T): T;
    tanh<T extends Tensor>(x: T): T;
    asinh<T extends Tensor>(x: T): T;
    acosh<T extends Tensor>(x: T): T;
    atanh<T extends Tensor>(x: T): T;
    erf<T extends Tensor>(x: T): T;
    step<T extends Tensor>(x: T, alpha: number): T;
    softmax<T extends Tensor>(logits: T, dim: number): T;
    logSoftmax<T extends Tensor>(logits: T, axis: number): T;
    image: {
        resizeBilinear<T extends Tensor3D | Tensor4D>(images: T, size: [number, number], alignCorners: boolean): T;
        resizeNearestNeighbor<T extends Tensor3D | Tensor4D>(images: T, size: [number, number], alignCorners: boolean): T;
    };
    unsortedSegmentSum<T extends Tensor>(x: T, segmentIds: Tensor1D | TensorLike1D, numSegments: number): T;
    topk<T extends Tensor>(x: T, k: number, sorted: boolean): {
        values: T;
        indices: T;
    };
    stridedSlice(x: Tensor, begin: number[], end: number[], strides: number[], beginMask: number, endMask: number, ellipsisMask: number, newAxisMask: number, shrinkAxisMask: number): Tensor;
    spectral: {
        fft(x: Tensor): Tensor;
        ifft(x: Tensor): Tensor;
        rfft(x: Tensor): Tensor;
        irfft(x: Tensor): Tensor;
    };
}
/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
export declare function setTensorTracker(fn: () => TensorTracker): void;
/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
export declare function setOpHandler(handler: OpHandler): void;
/**
 * Sets the deprecation warning function to be used by this file. This way the
 * Tensor class can be a leaf but still use the environment.
 */
export declare function setDeprecationWarningFn(fn: (msg: string) => void): void;
/**
 * We wrap data id since we use weak map to avoid memory leaks.
 * Since we have our own memory management, we have a reference counter
 * mapping a tensor to its data, so there is always a pointer (even if that
 * data is otherwise garbage collectable).
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
 * Global_Objects/WeakMap
 */
export declare type DataId = object;
export declare namespace Tensor { }
/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export declare class Tensor<R extends Rank = Rank> {
    /** Unique id of this tensor. */
    readonly id: number;
    /**
     * Id of the bucket holding the data for this tensor. Multiple arrays can
     * point to the same bucket (e.g. when calling array.reshape()).
     */
    dataId: DataId;
    /** The shape of the tensor. */
    readonly shape: ShapeMap[R];
    /** Number of elements in the tensor. */
    readonly size: number;
    /** The data type for the array. */
    readonly dtype: DataType;
    /** The rank type for the array (see `Rank` enum). */
    readonly rankType: R;
    /** Whether this tensor has been globally kept. */
    kept: boolean;
    /** The id of the scope this tensor is being tracked in. */
    scopeId: number;
    /**
     * Number of elements to skip in each dimension when indexing. See
     * https://docs.scipy.org/doc/numpy/reference/generated/\
     * numpy.ndarray.strides.html
     */
    readonly strides: number[];
    constructor(shape: ShapeMap[R], dtype: DataType, dataId: DataId, id: number);
    /** Flatten a Tensor to a 1D array. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    flatten(): Tensor1D;
    /** Converts a size-1 `tf.Tensor` to a `tf.Scalar`. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    asScalar(): Scalar;
    /** Converts a `tf.Tensor` to a `tf.Tensor1D`. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as1D(): Tensor1D;
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor2D`.
     *
     * @param rows Number of rows in `tf.Tensor2D`.
     * @param columns Number of columns in `tf.Tensor2D`.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as2D(rows: number, columns: number): Tensor2D;
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor3D`.
     *
     * @param rows Number of rows in `tf.Tensor3D`.
     * @param columns Number of columns in `tf.Tensor3D`.
     * @param depth Depth of `tf.Tensor3D`.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as3D(rows: number, columns: number, depth: number): Tensor3D;
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor4D`.
     *
     * @param rows Number of rows in `tf.Tensor4D`.
     * @param columns Number of columns in `tf.Tensor4D`.
     * @param depth Depth of `tf.Tensor4D`.
     * @param depth2 4th dimension of `tf.Tensor4D`.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as4D(rows: number, columns: number, depth: number, depth2: number): Tensor4D;
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor5D`.
     *
     * @param rows Number of rows in `tf.Tensor5D`.
     * @param columns Number of columns in `tf.Tensor5D`.
     * @param depth Depth of `tf.Tensor5D`.
     * @param depth2 4th dimension of `tf.Tensor5D`.
     * @param depth3 5th dimension of 'tf.Tensor5D'
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as5D(rows: number, columns: number, depth: number, depth2: number, depth3: number): Tensor5D;
    /**
     * Casts a `tf.Tensor` to a specified dtype.
     *
     * @param dtype Data-type to cast the tensor to.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    asType<T extends this>(this: T, dtype: DataType): T;
    readonly rank: number;
    /**
     * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    buffer<D extends DataType = 'float32'>(): Promise<TensorBuffer<R, D>>;
    /** Returns a `tf.TensorBuffer` that holds the underlying data. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    bufferSync<D extends DataType = 'float32'>(): TensorBuffer<R, D>;
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * asynchronously.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    array(): Promise<ArrayMap[R]>;
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * synchronously.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    arraySync(): ArrayMap[R];
    /**
     * Asynchronously downloads the values from the `tf.Tensor`. Returns a
     * promise of `TypedArray` that resolves when the computation has finished.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]>;
    /**
     * Synchronously downloads the values from the `tf.Tensor`. This blocks the
     * UI thread until the values are ready, which can cause performance issues.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D];
    /** Returns the underlying bytes of the tensor's data. */
    bytes(): Promise<Uint8Array[] | Uint8Array>;
    /**
     * Disposes `tf.Tensor` from memory.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    dispose(): void;
    protected isDisposedInternal: boolean;
    readonly isDisposed: boolean;
    throwIfDisposed(): void;
    /** Casts the array to type `float32` */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toFloat<T extends this>(this: T): T;
    /** Casts the array to type `int32` */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toInt(): this;
    /** Casts the array to type `bool` */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toBool(): this;
    /**
     * Prints the `tf.Tensor`. See `tf.print` for details.
     *
     * @param verbose Whether to print verbose information about the tensor,
     *    including dtype and size.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    print(verbose?: boolean): void;
    /**
     * Reshapes the tensor into the provided shape.
     * See `tf.reshape` for more details.
     *
     * @param newShape An array of integers defining the output tensor shape.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    reshape<R2 extends Rank>(newShape: ShapeMap[R2]): Tensor<R2>;
    /**
     * Reshapes the tensor into the shape of the provided tensor.
     *
     * @param x The tensor of required shape.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    reshapeAs<T extends Tensor>(x: T): T;
    /**
     * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
     * into the tensor's shape. See `tf.expandDims` for details.
     *
     * @param axis The dimension index at which to insert shape of 1. Defaults to
     *     0 (the first dimension).
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    expandDims<R2 extends Rank>(axis?: number): Tensor<R2>;
    /**
     * Returns a `tf.Tensor` with dimensions of size 1 removed from the shape.
     * See `tf.squeeze` for more details.
     *
     * @param axis A list of numbers. If specified, only squeezes the
     *    dimensions listed. The dimension index starts at 0. It is an error to
     *    squeeze a dimension that is not 1.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    squeeze<T extends Tensor>(axis?: number[]): T;
    /** Returns a copy of the tensor. See `tf.clone` for details. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    clone<T extends Tensor>(this: T): T;
    /**
     * Returns a human-readable description of the tensor. Useful for logging.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toString(verbose?: boolean): string;
    gather<T extends this>(this: T, indices: Tensor | TensorLike, axis?: number): T;
    norm(ord?: number | 'euclidean' | 'fro', axis?: number | number[], keepDims?: boolean): Tensor;
    slice<T extends Tensor<R>>(this: T, begin: number | number[], size?: number | number[]): T;
    reverse<T extends Tensor>(this: T, axis?: number | number[]): T;
    stack(x: Tensor, axis?: number): Tensor;
    unstack(axis?: number): Tensor[];
    all<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    any<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    logSumExp<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    sum<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    prod<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    mean<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    min<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    argMin<T extends Tensor>(axis?: number): T;
    argMax<T extends Tensor>(axis?: number): T;
    cast<T extends this>(dtype: DataType): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    addStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    subStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    powStrict(exp: Tensor | TensorLike): Tensor<R>;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    mulStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    divStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    minimumStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    maximumStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    modStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    squaredDifferenceStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    notEqualStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    lessStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    equalStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    lessEqualStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    greaterStrict<T extends this>(this: T, x: T | TensorLike): T;
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    greaterEqualStrict<T extends this>(this: T, x: T | TensorLike): T;
    logicalAnd(x: Tensor | TensorLike): Tensor;
    logicalOr(x: Tensor | TensorLike): Tensor;
    logicalNot<T extends Tensor>(this: T): T;
    logicalXor(x: Tensor | TensorLike): Tensor;
    where(condition: Tensor | TensorLike, x: Tensor | TensorLike): Tensor;
    neg<T extends Tensor>(this: T): T;
    ceil<T extends Tensor>(this: T): T;
    floor<T extends Tensor>(this: T): T;
    sign<T extends Tensor>(this: T): T;
    isNaN<T extends Tensor>(this: T): T;
    isInf<T extends Tensor>(this: T): T;
    isFinite<T extends Tensor>(this: T): T;
    exp<T extends Tensor>(this: T): T;
    expm1<T extends Tensor>(this: T): T;
    log<T extends Tensor>(this: T): T;
    log1p<T extends Tensor>(this: T): T;
    sqrt<T extends Tensor>(this: T): T;
    rsqrt<T extends Tensor>(this: T): T;
    square<T extends Tensor>(this: T): T;
    reciprocal<T extends Tensor>(this: T): T;
    abs<T extends Tensor>(this: T): T;
    clipByValue(min: number, max: number): Tensor<R>;
    sigmoid<T extends Tensor>(this: T): T;
    logSigmoid<T extends Tensor>(this: T): T;
    softplus<T extends Tensor>(this: T): T;
    zerosLike<T extends Tensor>(this: T): T;
    onesLike<T extends Tensor>(this: T): T;
    sin<T extends Tensor>(this: T): T;
    cos<T extends Tensor>(this: T): T;
    tan<T extends Tensor>(this: T): T;
    asin<T extends Tensor>(this: T): T;
    acos<T extends Tensor>(this: T): T;
    atan<T extends Tensor>(this: T): T;
    sinh<T extends Tensor>(this: T): T;
    cosh<T extends Tensor>(this: T): T;
    tanh<T extends Tensor>(this: T): T;
    asinh<T extends Tensor>(this: T): T;
    acosh<T extends Tensor>(this: T): T;
    atanh<T extends Tensor>(this: T): T;
    erf<T extends Tensor>(this: T): T;
    round<T extends Tensor>(this: T): T;
    step<T extends Tensor>(this: T, alpha?: number): T;
    softmax<T extends this>(this: T, dim?: number): T;
    logSoftmax<T extends this>(this: T, axis?: number): T;
    resizeBilinear<T extends Tensor3D | Tensor4D>(this: T, newShape2D: [number, number], alignCorners?: boolean): T;
    resizeNearestNeighbor<T extends Tensor3D | Tensor4D>(this: T, newShape2D: [number, number], alignCorners?: boolean): T;
    variable(trainable?: boolean, name?: string, dtype?: DataType): Variable<R>;
    unsortedSegmentSum<T extends Tensor>(this: T, segmentIds: Tensor1D | TensorLike1D, numSegments: number): T;
    topk<T extends Tensor>(this: T, k?: number, sorted?: boolean): {
        values: T;
        indices: T;
    };
    stridedSlice(this: Tensor, begin: number[], end: number[], strides: number[], beginMask?: number, endMask?: number, ellipsisMask?: number, newAxisMask?: number, shrinkAxisMask?: number): Tensor;
    fft(this: Tensor): Tensor;
    ifft(this: Tensor): Tensor;
    rfft(this: Tensor): Tensor;
    irfft(this: Tensor): Tensor;
}
export interface NumericTensor<R extends Rank = Rank> extends Tensor<R> {
    dtype: NumericDataType;
    dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D];
    data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]>;
}
export interface StringTensor<R extends Rank = Rank> extends Tensor<R> {
    dtype: 'string';
    dataSync<D extends DataType = 'string'>(): DataTypeMap[D];
    data<D extends DataType = 'string'>(): Promise<DataTypeMap[D]>;
}
/** @doclink Tensor */
export declare type Scalar = Tensor<Rank.R0>;
/** @doclink Tensor */
export declare type Tensor1D = Tensor<Rank.R1>;
/** @doclink Tensor */
export declare type Tensor2D = Tensor<Rank.R2>;
/** @doclink Tensor */
export declare type Tensor3D = Tensor<Rank.R3>;
/** @doclink Tensor */
export declare type Tensor4D = Tensor<Rank.R4>;
/** @doclink Tensor */
export declare type Tensor5D = Tensor<Rank.R5>;
/** @doclink Tensor */
export declare type Tensor6D = Tensor<Rank.R6>;
/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export declare class Variable<R extends Rank = Rank> extends Tensor<R> {
    trainable: boolean;
    name: string;
    constructor(initialValue: Tensor<R>, trainable: boolean, name: string, tensorId: number);
    /**
     * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
     * the same shape and dtype as the old `tf.Tensor`.
     *
     * @param newValue New tensor to be assigned to this variable.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    assign(newValue: Tensor<R>): void;
    dispose(): void;
}

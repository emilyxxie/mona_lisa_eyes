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
import { tensorToString } from './tensor_format';
import * as util from './util';
import { computeStrides, toNestedArray } from './util';
/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class TensorBuffer {
    constructor(shape, dtype, values) {
        this.dtype = dtype;
        this.shape = shape.slice();
        this.size = util.sizeFromShape(shape);
        if (values != null) {
            const n = values.length;
            util.assert(n === this.size, () => `Length of values '${n}' does not match the size ` +
                `inferred by the shape '${this.size}'.`);
        }
        if (dtype === 'complex64') {
            throw new Error(`complex64 dtype TensorBuffers are not supported. Please create ` +
                `a TensorBuffer for the real and imaginary parts separately and ` +
                `call tf.complex(real, imag).`);
        }
        this.values = values || util.getArrayFromDType(dtype, this.size);
        this.strides = computeStrides(shape);
    }
    /**
     * Sets a value in the buffer at a given location.
     *
     * @param value The value to set.
     * @param locs  The location indices.
     */
    /** @doc {heading: 'Tensors', subheading: 'Creation'} */
    set(value, ...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        util.assert(locs.length === this.rank, () => `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
        const index = this.locToIndex(locs);
        this.values[index] = value;
    }
    /**
     * Returns the value in the buffer at the provided location.
     *
     * @param locs The location indices.
     */
    /** @doc {heading: 'Tensors', subheading: 'Creation'} */
    get(...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        let i = 0;
        for (const loc of locs) {
            if (loc < 0 || loc >= this.shape[i]) {
                const msg = `Requested out of range element at ${locs}. ` +
                    `  Buffer shape=${this.shape}`;
                throw new Error(msg);
            }
            i++;
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.values[index];
    }
    locToIndex(locs) {
        if (this.rank === 0) {
            return 0;
        }
        else if (this.rank === 1) {
            return locs[0];
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    }
    indexToLoc(index) {
        if (this.rank === 0) {
            return [];
        }
        else if (this.rank === 1) {
            return [index];
        }
        const locs = new Array(this.shape.length);
        for (let i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Creates an immutable `tf.Tensor` object from the buffer.
     */
    /** @doc {heading: 'Tensors', subheading: 'Creation'} */
    toTensor() {
        return trackerFn().makeTensor(this.values, this.shape, this.dtype);
    }
}
// For tracking tensor creation and disposal.
let trackerFn = null;
// Used by chaining methods to call into ops.
let opHandler = null;
// Used to warn about deprecated methods.
let deprecationWarningFn = null;
// This here so that we can use this method on dev branches and keep the
// functionality at master.
// tslint:disable-next-line:no-unused-expression
[deprecationWarningFn];
/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
export function setTensorTracker(fn) {
    trackerFn = fn;
}
/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
export function setOpHandler(handler) {
    opHandler = handler;
}
/**
 * Sets the deprecation warning function to be used by this file. This way the
 * Tensor class can be a leaf but still use the environment.
 */
export function setDeprecationWarningFn(fn) {
    deprecationWarningFn = fn;
}
/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class Tensor {
    constructor(shape, dtype, dataId, id) {
        /** Whether this tensor has been globally kept. */
        this.kept = false;
        this.isDisposedInternal = false;
        this.shape = shape.slice();
        this.dtype = dtype || 'float32';
        this.size = util.sizeFromShape(shape);
        this.strides = computeStrides(shape);
        this.dataId = dataId;
        this.id = id;
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
    }
    /** Flatten a Tensor to a 1D array. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    flatten() {
        this.throwIfDisposed();
        return this.as1D();
    }
    /** Converts a size-1 `tf.Tensor` to a `tf.Scalar`. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    asScalar() {
        this.throwIfDisposed();
        util.assert(this.size === 1, () => 'The array must have only 1 element.');
        return this.reshape([]);
    }
    /** Converts a `tf.Tensor` to a `tf.Tensor1D`. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as1D() {
        this.throwIfDisposed();
        return this.reshape([this.size]);
    }
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor2D`.
     *
     * @param rows Number of rows in `tf.Tensor2D`.
     * @param columns Number of columns in `tf.Tensor2D`.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as2D(rows, columns) {
        this.throwIfDisposed();
        return this.reshape([rows, columns]);
    }
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor3D`.
     *
     * @param rows Number of rows in `tf.Tensor3D`.
     * @param columns Number of columns in `tf.Tensor3D`.
     * @param depth Depth of `tf.Tensor3D`.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as3D(rows, columns, depth) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth]);
    }
    /**
     * Converts a `tf.Tensor` to a `tf.Tensor4D`.
     *
     * @param rows Number of rows in `tf.Tensor4D`.
     * @param columns Number of columns in `tf.Tensor4D`.
     * @param depth Depth of `tf.Tensor4D`.
     * @param depth2 4th dimension of `tf.Tensor4D`.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    as4D(rows, columns, depth, depth2) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth, depth2]);
    }
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
    as5D(rows, columns, depth, depth2, depth3) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth, depth2, depth3]);
    }
    /**
     * Casts a `tf.Tensor` to a specified dtype.
     *
     * @param dtype Data-type to cast the tensor to.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    asType(dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    async buffer() {
        const vals = await this.data();
        return opHandler.buffer(this.shape, this.dtype, vals);
    }
    /** Returns a `tf.TensorBuffer` that holds the underlying data. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    bufferSync() {
        return opHandler.buffer(this.shape, this.dtype, this.dataSync());
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * asynchronously.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    async array() {
        const vals = await this.data();
        return toNestedArray(this.shape, vals);
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * synchronously.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    arraySync() {
        return toNestedArray(this.shape, this.dataSync());
    }
    /**
     * Asynchronously downloads the values from the `tf.Tensor`. Returns a
     * promise of `TypedArray` that resolves when the computation has finished.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    async data() {
        this.throwIfDisposed();
        const data = trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            const bytes = await data;
            try {
                return bytes.map(b => util.decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /**
     * Synchronously downloads the values from the `tf.Tensor`. This blocks the
     * UI thread until the values are ready, which can cause performance issues.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    dataSync() {
        this.throwIfDisposed();
        const data = trackerFn().readSync(this.dataId);
        if (this.dtype === 'string') {
            try {
                return data.map(b => util.decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /** Returns the underlying bytes of the tensor's data. */
    async bytes() {
        this.throwIfDisposed();
        const data = await trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            return data;
        }
        else {
            return new Uint8Array(data.buffer);
        }
    }
    /**
     * Disposes `tf.Tensor` from memory.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    dispose() {
        if (this.isDisposed) {
            return;
        }
        trackerFn().disposeTensor(this);
        this.isDisposedInternal = true;
    }
    get isDisposed() {
        return this.isDisposedInternal;
    }
    throwIfDisposed() {
        if (this.isDisposed) {
            throw new Error(`Tensor is disposed.`);
        }
    }
    /** Casts the array to type `float32` */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toFloat() {
        return this.asType('float32');
    }
    /** Casts the array to type `int32` */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toInt() {
        return this.asType('int32');
    }
    /** Casts the array to type `bool` */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toBool() {
        return this.asType('bool');
    }
    /**
     * Prints the `tf.Tensor`. See `tf.print` for details.
     *
     * @param verbose Whether to print verbose information about the tensor,
     *    including dtype and size.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    print(verbose = false) {
        return opHandler.print(this, verbose);
    }
    /**
     * Reshapes the tensor into the provided shape.
     * See `tf.reshape` for more details.
     *
     * @param newShape An array of integers defining the output tensor shape.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    reshape(newShape) {
        this.throwIfDisposed();
        return opHandler.reshape(this, newShape);
    }
    /**
     * Reshapes the tensor into the shape of the provided tensor.
     *
     * @param x The tensor of required shape.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    reshapeAs(x) {
        this.throwIfDisposed();
        return this.reshape(x.shape);
    }
    /**
     * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
     * into the tensor's shape. See `tf.expandDims` for details.
     *
     * @param axis The dimension index at which to insert shape of 1. Defaults to
     *     0 (the first dimension).
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    expandDims(axis = 0) {
        return opHandler.expandDims(this, axis);
    }
    /**
     * Returns a `tf.Tensor` with dimensions of size 1 removed from the shape.
     * See `tf.squeeze` for more details.
     *
     * @param axis A list of numbers. If specified, only squeezes the
     *    dimensions listed. The dimension index starts at 0. It is an error to
     *    squeeze a dimension that is not 1.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    squeeze(axis) {
        this.throwIfDisposed();
        return opHandler.squeeze(this, axis);
    }
    /** Returns a copy of the tensor. See `tf.clone` for details. */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    clone() {
        this.throwIfDisposed();
        return opHandler.clone(this);
    }
    /**
     * Returns a human-readable description of the tensor. Useful for logging.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    toString(verbose = false) {
        const vals = this.dataSync();
        return tensorToString(vals, this.shape, this.dtype, verbose);
    }
    // Below is chain API that is not exposed to docs to avoid repetition. To
    // expose a method, move it above this comment and add @doc and jsdoc.
    gather(indices, axis = 0) {
        this.throwIfDisposed();
        return opHandler.gather(this, indices, axis);
    }
    norm(ord = 'euclidean', axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.norm(this, ord, axis, keepDims);
    }
    slice(begin, size) {
        this.throwIfDisposed();
        return opHandler.slice(this, begin, size);
    }
    reverse(axis) {
        this.throwIfDisposed();
        return opHandler.reverse(this, axis);
    }
    stack(x, axis = 0) {
        return opHandler.stack([this, x], axis);
    }
    unstack(axis = 0) {
        return opHandler.unstack(this, axis);
    }
    // Reduction ops.
    all(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.all(this, axis, keepDims);
    }
    any(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.any(this, axis, keepDims);
    }
    logSumExp(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.logSumExp(this, axis, keepDims);
    }
    sum(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.sum(this, axis, keepDims);
    }
    prod(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.prod(this, axis, keepDims);
    }
    mean(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.mean(this, axis, keepDims);
    }
    min(axis = null, keepDims = false) {
        this.throwIfDisposed();
        return opHandler.min(this, axis, keepDims);
    }
    argMin(axis = null) {
        this.throwIfDisposed();
        return opHandler.argMin(this, axis);
    }
    argMax(axis = null) {
        this.throwIfDisposed();
        return opHandler.argMax(this, axis);
    }
    // Transformations
    cast(dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    }
    // Binary ops.
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    addStrict(x) {
        this.throwIfDisposed();
        return opHandler.addStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    subStrict(x) {
        this.throwIfDisposed();
        return opHandler.subStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    powStrict(exp) {
        this.throwIfDisposed();
        return opHandler.powStrict(this, exp);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    mulStrict(x) {
        this.throwIfDisposed();
        return opHandler.mulStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    divStrict(x) {
        this.throwIfDisposed();
        return opHandler.divStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    minimumStrict(x) {
        this.throwIfDisposed();
        return opHandler.minimumStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    maximumStrict(x) {
        this.throwIfDisposed();
        return opHandler.maximumStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    modStrict(x) {
        this.throwIfDisposed();
        return opHandler.modStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    squaredDifferenceStrict(x) {
        this.throwIfDisposed();
        return opHandler.squaredDifferenceStrict(this, x);
    }
    // Compare ops.
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    notEqualStrict(x) {
        this.throwIfDisposed();
        return opHandler.notEqualStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    lessStrict(x) {
        this.throwIfDisposed();
        return opHandler.lessStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    equalStrict(x) {
        this.throwIfDisposed();
        return opHandler.equalStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    lessEqualStrict(x) {
        this.throwIfDisposed();
        return opHandler.lessEqualStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    greaterStrict(x) {
        this.throwIfDisposed();
        return opHandler.greaterStrict(this, x);
    }
    /**
     * @deprecated strict variants of ops have been deprecated
     */
    greaterEqualStrict(x) {
        this.throwIfDisposed();
        return opHandler.greaterEqualStrict(this, x);
    }
    // Compare ops.
    logicalAnd(x) {
        this.throwIfDisposed();
        return opHandler.logicalAnd(this, x);
    }
    logicalOr(x) {
        this.throwIfDisposed();
        return opHandler.logicalOr(this, x);
    }
    logicalNot() {
        this.throwIfDisposed();
        return opHandler.logicalNot(this);
    }
    logicalXor(x) {
        this.throwIfDisposed();
        return opHandler.logicalXor(this, x);
    }
    where(condition, x) {
        this.throwIfDisposed();
        return opHandler.where(condition, this, x);
    }
    // Unary ops.
    neg() {
        this.throwIfDisposed();
        return opHandler.neg(this);
    }
    ceil() {
        this.throwIfDisposed();
        return opHandler.ceil(this);
    }
    floor() {
        this.throwIfDisposed();
        return opHandler.floor(this);
    }
    sign() {
        this.throwIfDisposed();
        return opHandler.sign(this);
    }
    isNaN() {
        this.throwIfDisposed();
        return opHandler.isNaN(this);
    }
    isInf() {
        this.throwIfDisposed();
        return opHandler.isInf(this);
    }
    isFinite() {
        this.throwIfDisposed();
        return opHandler.isFinite(this);
    }
    exp() {
        this.throwIfDisposed();
        return opHandler.exp(this);
    }
    expm1() {
        this.throwIfDisposed();
        return opHandler.expm1(this);
    }
    log() {
        this.throwIfDisposed();
        return opHandler.log(this);
    }
    log1p() {
        this.throwIfDisposed();
        return opHandler.log1p(this);
    }
    sqrt() {
        this.throwIfDisposed();
        return opHandler.sqrt(this);
    }
    rsqrt() {
        this.throwIfDisposed();
        return opHandler.rsqrt(this);
    }
    square() {
        this.throwIfDisposed();
        return opHandler.square(this);
    }
    reciprocal() {
        this.throwIfDisposed();
        return opHandler.reciprocal(this);
    }
    abs() {
        this.throwIfDisposed();
        return opHandler.abs(this);
    }
    clipByValue(min, max) {
        this.throwIfDisposed();
        return opHandler.clipByValue(this, min, max);
    }
    sigmoid() {
        this.throwIfDisposed();
        return opHandler.sigmoid(this);
    }
    logSigmoid() {
        this.throwIfDisposed();
        return opHandler.logSigmoid(this);
    }
    softplus() {
        this.throwIfDisposed();
        return opHandler.softplus(this);
    }
    zerosLike() {
        this.throwIfDisposed();
        return opHandler.zerosLike(this);
    }
    onesLike() {
        this.throwIfDisposed();
        return opHandler.onesLike(this);
    }
    sin() {
        this.throwIfDisposed();
        return opHandler.sin(this);
    }
    cos() {
        this.throwIfDisposed();
        return opHandler.cos(this);
    }
    tan() {
        this.throwIfDisposed();
        return opHandler.tan(this);
    }
    asin() {
        this.throwIfDisposed();
        return opHandler.asin(this);
    }
    acos() {
        this.throwIfDisposed();
        return opHandler.acos(this);
    }
    atan() {
        this.throwIfDisposed();
        return opHandler.atan(this);
    }
    sinh() {
        this.throwIfDisposed();
        return opHandler.sinh(this);
    }
    cosh() {
        this.throwIfDisposed();
        return opHandler.cosh(this);
    }
    tanh() {
        this.throwIfDisposed();
        return opHandler.tanh(this);
    }
    asinh() {
        this.throwIfDisposed();
        return opHandler.asinh(this);
    }
    acosh() {
        this.throwIfDisposed();
        return opHandler.acosh(this);
    }
    atanh() {
        this.throwIfDisposed();
        return opHandler.atanh(this);
    }
    erf() {
        this.throwIfDisposed();
        return opHandler.erf(this);
    }
    round() {
        this.throwIfDisposed();
        return opHandler.round(this);
    }
    step(alpha = 0.0) {
        this.throwIfDisposed();
        return opHandler.step(this, alpha);
    }
    softmax(dim = -1) {
        this.throwIfDisposed();
        return opHandler.softmax(this, dim);
    }
    logSoftmax(axis = -1) {
        this.throwIfDisposed();
        return opHandler.logSoftmax(this, axis);
    }
    // Image ops.
    resizeBilinear(newShape2D, alignCorners = false) {
        this.throwIfDisposed();
        return opHandler.image.resizeBilinear(this, newShape2D, alignCorners);
    }
    resizeNearestNeighbor(newShape2D, alignCorners = false) {
        this.throwIfDisposed();
        return opHandler.image.resizeNearestNeighbor(this, newShape2D, alignCorners);
    }
    // Pooling.
    variable(trainable = true, name, dtype) {
        this.throwIfDisposed();
        return trackerFn().makeVariable(this, trainable, name, dtype);
    }
    unsortedSegmentSum(segmentIds, numSegments) {
        this.throwIfDisposed();
        return opHandler.unsortedSegmentSum(this, segmentIds, numSegments);
    }
    topk(k = 1, sorted = true) {
        this.throwIfDisposed();
        return opHandler.topk(this, k, sorted);
    }
    stridedSlice(begin, end, strides, beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0, shrinkAxisMask = 0) {
        this.throwIfDisposed();
        return opHandler.stridedSlice(this, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }
    fft() {
        this.throwIfDisposed();
        return opHandler.spectral.fft(this);
    }
    ifft() {
        this.throwIfDisposed();
        return opHandler.spectral.ifft(this);
    }
    rfft() {
        this.throwIfDisposed();
        return opHandler.spectral.rfft(this);
    }
    irfft() {
        this.throwIfDisposed();
        return opHandler.spectral.irfft(this);
    }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
    value: (instance) => {
        return !!instance && instance.dataId != null && instance.shape != null &&
            instance.dtype != null;
    }
});
/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class Variable extends Tensor {
    constructor(initialValue, trainable, name, tensorId) {
        super(initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
        this.trainable = trainable;
        this.name = name;
    }
    /**
     * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
     * the same shape and dtype as the old `tf.Tensor`.
     *
     * @param newValue New tensor to be assigned to this variable.
     */
    /** @doc {heading: 'Tensors', subheading: 'Classes'} */
    assign(newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error(`dtype of the new value (${newValue.dtype}) and ` +
                `previous value (${this.dtype}) must match`);
        }
        if (!util.arraysEqual(newValue.shape, this.shape)) {
            throw new Error(`shape of the new value (${newValue.shape}) and ` +
                `previous value (${this.shape}) must match`);
        }
        trackerFn().disposeTensor(this);
        this.dataId = newValue.dataId;
        trackerFn().incRef(this, null /* backend */);
    }
    dispose() {
        trackerFn().disposeVariable(this);
        this.isDisposedInternal = true;
    }
}
Object.defineProperty(Variable, Symbol.hasInstance, {
    value: (instance) => {
        return instance instanceof Tensor && instance.assign != null &&
            instance.assign instanceof Function;
    }
});
//# sourceMappingURL=tensor.js.map
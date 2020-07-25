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
import { Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D } from '../tensor';
import { Rank, TensorLike } from '../types';
/**
 * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
 * of length `size`. See `slice` for details.
 */
declare function slice1d_(x: Tensor1D | TensorLike, begin: number, size: number): Tensor1D;
/**
 * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
declare function slice2d_(x: Tensor2D | TensorLike, begin: [number, number], size: [number, number]): Tensor2D;
/**
 * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
declare function slice3d_(x: Tensor3D | TensorLike, begin: [number, number, number], size: [number, number, number]): Tensor3D;
/**
 * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
declare function slice4d_(x: Tensor4D | TensorLike, begin: [number, number, number, number], size: [number, number, number, number]): Tensor4D;
/**
 * Extracts a slice from a `tf.Tensor` starting at coordinates `begin`
 * and is of size `size`.
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `x` is of the given rank:
 *   - `tf.slice1d`
 *   - `tf.slice2d`
 *   - `tf.slice3d`
 *   - `tf.slice4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.slice([1], [2]).print();
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * x.slice([1, 0], [1, 2]).print();
 * ```
 * @param x The input `tf.Tensor` to slice from.
 * @param begin The coordinates to start the slice from. The length can be
 *     less than the rank of x - the rest of the axes will have implicit 0 as
 *     start. Can also be a single number, in which case it specifies the
 *     first axis.
 * @param size The size of the slice. The length can be less than the rank of
 *     x - the rest of the axes will have implicit -1. A value of -1 requests
 *     the rest of the dimensions in the axis. Can also be a single number,
 *     in which case it specifies the size of the first axis.
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
declare function slice_<R extends Rank, T extends Tensor<R>>(x: T | TensorLike, begin: number | number[], size?: number | number[]): T;
export declare const slice: typeof slice_;
export declare const slice1d: typeof slice1d_;
export declare const slice2d: typeof slice2d_;
export declare const slice3d: typeof slice3d_;
export declare const slice4d: typeof slice4d_;
export {};

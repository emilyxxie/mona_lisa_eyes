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
import { ENGINE } from '../engine';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
import { scalar, zerosLike } from './tensor_ops';
/**
 * Computes `-1 * x` element-wise.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);
 *
 * x.neg().print();  // or tf.neg(x)
 * ```
 *
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function neg_(x) {
    const $x = convertToTensor(x, 'x', 'neg');
    const grad = (dy) => {
        return { x: () => dy.neg() };
    };
    const attrs = {};
    const inputsToSave = [$x];
    return ENGINE.runKernelFunc(backend => backend.neg($x), { x: $x }, grad, 'Neg', attrs, inputsToSave);
}
/**
 * Computes ceiling of input `tf.Tensor` element-wise: `ceil(x)`
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.ceil().print();  // or tf.ceil(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function ceil_(x) {
    const $x = convertToTensor(x, 'x', 'ceil');
    // TODO(manrajgrover): Return null for gradients when backprop supports it.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.ceil($x), { $x }, grad);
}
/**
 * Computes floor of input `tf.Tensor` element-wise: `floor(x)`.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.floor().print();  // or tf.floor(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function floor_(x) {
    const $x = convertToTensor(x, 'x', 'floor');
    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.floor($x), { $x }, grad);
}
/**
 * Returns an element-wise indication of the sign of a number.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);
 *
 * x.sign().print();  // or tf.sign(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sign_(x) {
    const $x = convertToTensor(x, 'x', 'sign');
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.sign($x), { $x }, grad);
}
/**
 * RReturns which elements of x are NaN.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isNaN().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function isNaN_(x) {
    const $x = convertToTensor(x, 'x', 'isNaN');
    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.isNaN($x), { $x }, grad);
}
/**
 * Returns which elements of x are Infinity or -Infinity.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isInf().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function isInf_(x) {
    const $x = convertToTensor(x, 'x', 'isInf');
    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.isInf($x), { $x }, grad);
}
/**
 * Returns which elements of x are finite.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isFinite().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function isFinite_(x) {
    const $x = convertToTensor(x, 'x', 'isFinite');
    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.isFinite($x), { $x }, grad);
}
/**
 * Computes round of input `tf.Tensor` element-wise: `round(x)`.
 * It implements banker's rounding.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.round().print();  // or tf.round(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function round_(x) {
    const $x = convertToTensor(x, 'x', 'round');
    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.round($x), { $x }, grad);
}
/**
 * Computes exponential of the input `tf.Tensor` element-wise. `e ^ x`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, -3]);
 *
 * x.exp().print();  // or tf.exp(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function exp_(x) {
    const $x = convertToTensor(x, 'x', 'exp');
    const bck = (dy, saved) => {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        return { x: () => dy.mul(saved[0]) };
    };
    const attrs = {};
    const inputsToSave = [];
    const outputsToSave = [true];
    return ENGINE.runKernelFunc((backend, save) => {
        const y = backend.exp($x);
        save([y]);
        return y;
    }, { x: $x }, bck, 'Exp', attrs, inputsToSave, outputsToSave);
}
/**
 * Computes exponential of the input `tf.Tensor` minus one element-wise.
 * `e ^ x - 1`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, -3]);
 *
 * x.expm1().print();  // or tf.expm1(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function expm1_(x) {
    const $x = convertToTensor(x, 'x', 'expm1');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.mul($x.exp()) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.expm1($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes natural logarithm of the input `tf.Tensor` element-wise: `ln(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.E]);
 *
 * x.log().print();  // or tf.log(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function log_(x) {
    const $x = convertToTensor(x, 'x', 'log');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { x: () => dy.div($x.toFloat()) };
    };
    const attrs = {};
    const inputsToSave = [$x];
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.log($x);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'Log', attrs, inputsToSave);
}
/**
 * Computes natural logarithm of the input `tf.Tensor` plus one
 * element-wise: `ln(1 + x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.E - 1]);
 *
 * x.log1p().print();  // or tf.log1p(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function log1p_(x) {
    const $x = convertToTensor(x, 'x', 'log1p');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.div($x.add(1)) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.log1p($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes square root of the input `tf.Tensor` element-wise: `y = sqrt(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 4, -1]);
 *
 * x.sqrt().print();  // or tf.sqrt(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sqrt_(x) {
    const $x = convertToTensor(x, 'x', 'sqrt');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { x: () => dy.div($x.toFloat().sqrt().mul(2)) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.sqrt($x);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'Sqrt', {});
}
/**
 * Computes reciprocal of square root of the input `tf.Tensor` element-wise:
 * `y = 1 / sqrt(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 4, -1]);
 *
 * x.rsqrt().print();  // or tf.rsqrt(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function rsqrt_(x) {
    const $x = convertToTensor(x, 'x', 'rsqrt');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { x: () => dy.div($x.pow(1.5).mul(2)).neg() };
    };
    const inputsToSave = [$x];
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.rsqrt($x);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'Rsqrt', {} /* attrs */, inputsToSave);
}
/**
 * Computes reciprocal of x element-wise: `1 / x`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, 2]);
 *
 * x.reciprocal().print();  // or tf.reciprocal(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function reciprocal_(x) {
    const $x = convertToTensor(x, 'x', 'reciprocal');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.div($x.square().neg()) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.reciprocal($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes absolute value element-wise: `abs(x)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.abs().print();  // or tf.abs(x)
 * ```
 * @param x The input `tf.Tensor`.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function abs_(x) {
    const $x = convertToTensor(x, 'x', 'abs');
    if ($x.dtype === 'complex64') {
        return ENGINE.runKernelFunc(backend => backend.complexAbs($x), { $x });
    }
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { x: () => dy.mul($x.toFloat().step(-1)) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.abs($x);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'Abs');
}
/**
 * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
 * ```
 * @param x The input tensor.
 * @param clipValueMin Lower-bound of range to be clipped to.
 * @param clipValueMax Upper-bound of range to be clipped to.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function clipByValue_(x, clipValueMin, clipValueMax) {
    const $x = convertToTensor(x, 'x', 'clipByValue');
    util.assert((clipValueMin <= clipValueMax), () => `Error in clip: min (${clipValueMin}) must be ` +
        `less than or equal to max (${clipValueMax}).`);
    const grad = (dy, saved) => {
        const [$x] = saved;
        return {
            x: () => dy.where($x.greaterEqual(clipValueMin)
                .logicalAnd($x.lessEqual(clipValueMax)), zerosLike(dy)),
        };
    };
    const inputsToSave = [$x];
    const attr = { min: clipValueMin, max: clipValueMax };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.clip($x, clipValueMin, clipValueMax);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'ClipByValue', attr, inputsToSave);
}
/**
 * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
 *
 * ```js
 * const x = tf.tensor1d([0, -1, 2, -3]);
 *
 * x.sigmoid().print();  // or tf.sigmoid(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sigmoid_(x) {
    const $x = convertToTensor(x, 'x', 'sigmoid');
    const grad = (dy, saved) => {
        const [y] = saved;
        return { x: () => dy.mul(y.mul(scalar(1).sub(y))) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const y = backend.sigmoid($x);
        save([y]);
        return y;
    }, { x: $x }, grad, 'Sigmoid');
}
/**
 * Computes log sigmoid of the input `tf.Tensor` element-wise:
 * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.logSigmoid().print();  // or tf.logSigmoid(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function logSigmoid_(x) {
    const $x = convertToTensor(x, 'x', 'logSigmoid');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.mul($x.neg().sigmoid()) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.softplus($x.neg()).neg();
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes softplus of the input `tf.Tensor` element-wise: `log(exp(x) + 1)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.softplus().print();  // or tf.softplus(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function softplus_(x) {
    const $x = convertToTensor(x, 'x', 'softplus');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.mul($x.sigmoid()) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.softplus($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes sin of the input Tensor element-wise: `sin(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.sin().print();  // or tf.sin(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sin_(x) {
    const $x = convertToTensor(x, 'x', 'sin');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { x: () => $x.toFloat().cos().mul(dy) };
    };
    const inputsToSave = [$x];
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.sin($x);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'Sin', {} /* attrs */, inputsToSave);
}
/**
 * Computes cos of the input `tf.Tensor` element-wise: `cos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.cos().print();  // or tf.cos(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function cos_(x) {
    const $x = convertToTensor(x, 'x', 'cos');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { x: () => $x.toFloat().sin().neg().mul(dy) };
    };
    const inputsToSave = [$x];
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.cos($x);
        save([$x]);
        return res;
    }, { x: $x }, grad, 'Cos', {} /* attrs */, inputsToSave);
}
/**
 * Computes tan of the input `tf.Tensor` element-wise, `tan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.tan().print();  // or tf.tan(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function tan_(x) {
    const $x = convertToTensor(x, 'x', 'tan');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.div($x.cos().square()) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.tan($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes asin of the input `tf.Tensor` element-wise: `asin(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asin().print();  // or tf.asin(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function asin_(x) {
    const $x = convertToTensor(x, 'x', 'asin');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return {
            // tslint:disable-next-line: no-unnecessary-type-assertion
            $x: () => dy.div(scalar(1).sub($x.toFloat().square()).sqrt())
        };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.asin($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes acos of the input `tf.Tensor` element-wise: `acos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.acos().print();  // or tf.acos(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function acos_(x) {
    const $x = convertToTensor(x, 'x', 'acos');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return {
            $x: () => {
                const a = $x.toFloat().square();
                const b = scalar(1).sub(a).sqrt();
                // tslint:disable-next-line: no-unnecessary-type-assertion
                return dy.div(b).neg();
            }
        };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.acos($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes atan of the input `tf.Tensor` element-wise: `atan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.atan().print();  // or tf.atan(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function atan_(x) {
    const $x = convertToTensor(x, 'x', 'atan');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.div($x.toFloat().square().add(1)) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.atan($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes hyperbolic sin of the input `tf.Tensor` element-wise: `sinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.sinh().print();  // or tf.sinh(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sinh_(x) {
    const $x = convertToTensor(x, 'x', 'sinh');
    const grad = (dy, saved) => {
        const [$x] = saved;
        // tslint:disable-next-line: no-unnecessary-type-assertion
        return { $x: () => $x.toFloat().cosh().mul(dy) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.sinh($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes hyperbolic cos of the input `tf.Tensor` element-wise: `cosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.cosh().print();  // or tf.cosh(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function cosh_(x) {
    const $x = convertToTensor(x, 'x', 'cosh');
    const grad = (dy, saved) => {
        const [$x] = saved;
        // tslint:disable-next-line: no-unnecessary-type-assertion
        return { $x: () => $x.toFloat().sinh().mul(dy) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.cosh($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes hyperbolic tangent of the input `tf.Tensor` element-wise: `tanh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, 70]);
 *
 * x.tanh().print();  // or tf.tanh(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function tanh_(x) {
    const $x = convertToTensor(x, 'x', 'tanh');
    const grad = (dy, saved) => {
        const [y] = saved;
        // tslint:disable-next-line: no-unnecessary-type-assertion
        return { x: () => scalar(1).sub(y.square()).mul(dy) };
    };
    const outputsToSave = [true];
    return ENGINE.runKernelFunc((backend, save) => {
        const y = backend.tanh($x);
        save([y]);
        return y;
    }, { x: $x }, grad, 'Tanh', {} /* attrs */, null /* inputsToSave */, outputsToSave);
}
/**
 * Computes inverse hyperbolic sin of the input `tf.Tensor` element-wise:
 * `asinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asinh().print();  // or tf.asinh(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function asinh_(x) {
    const $x = convertToTensor(x, 'x', 'asinh');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return {
            $x: () => {
                const a = scalar(1).add($x.toFloat().square()).sqrt();
                // tslint:disable-next-line: no-unnecessary-type-assertion
                return dy.div(a);
            }
        };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.asinh($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes the inverse hyperbolic cos of the input `tf.Tensor` element-wise:
 * `acosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([10, 1, 3, 5.7]);
 *
 * x.acosh().print();  // or tf.acosh(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function acosh_(x) {
    const $x = convertToTensor(x, 'x', 'acosh');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return {
            $x: () => {
                const a = $x.toFloat().square().sub(1).sqrt();
                // tslint:disable-next-line: no-unnecessary-type-assertion
                return dy.div(a);
            }
        };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.acosh($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes inverse hyperbolic tan of the input `tf.Tensor` element-wise:
 * `atanh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.atanh().print();  // or tf.atanh(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function atanh_(x) {
    const $x = convertToTensor(x, 'x', 'atanh');
    const grad = (dy, saved) => {
        const [$x] = saved;
        return { $x: () => dy.div(scalar(1).sub($x.toFloat().square())) };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.atanh($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes gause error function of the input `tf.Tensor` element-wise:
 * `erf(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.erf().print(); // or tf.erf(x);
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function erf_(x) {
    let $x = convertToTensor(x, 'x', 'erf');
    util.assert($x.dtype === 'int32' || $x.dtype === 'float32', () => 'Input dtype must be `int32` or `float32`.');
    if ($x.dtype === 'int32') {
        $x = $x.toFloat();
    }
    const grad = (dy, saved) => {
        const [$x] = saved;
        return {
            $x: () => dy.mul($x.square().neg().exp().mul(2 / Math.sqrt(Math.PI)))
        };
    };
    return ENGINE.runKernelFunc((backend, save) => {
        const res = backend.erf($x);
        save([$x]);
        return res;
    }, { $x }, grad);
}
/**
 * Computes step of the input `tf.Tensor` element-wise: `x > 0 ? 1 : alpha * x`
 *
 * ```js
 * const x = tf.tensor1d([0, 2, -1, -3]);
 *
 * x.step(.5).print();  // or tf.step(x, .5)
 * ```
 * @param x The input tensor.
 * @param alpha The gradient when input is negative.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function step_(x, alpha = 0.0) {
    const $x = convertToTensor(x, 'x', 'step');
    // TODO(manrajgrover): Return null for gradients when backprop supports
    // it.
    const grad = (dy) => {
        return { $x: () => zerosLike(dy) };
    };
    return ENGINE.runKernelFunc(backend => backend.step($x, alpha), { $x }, grad);
}
export const abs = op({ abs_ });
export const acos = op({ acos_ });
export const acosh = op({ acosh_ });
export const asin = op({ asin_ });
export const asinh = op({ asinh_ });
export const atan = op({ atan_ });
export const atanh = op({ atanh_ });
export const ceil = op({ ceil_ });
export const clipByValue = op({ clipByValue_ });
export const cos = op({ cos_ });
export const cosh = op({ cosh_ });
export const erf = op({ erf_ });
export const exp = op({ exp_ });
export const expm1 = op({ expm1_ });
export const floor = op({ floor_ });
export const log = op({ log_ });
export const log1p = op({ log1p_ });
export const logSigmoid = op({ logSigmoid_ });
export const neg = op({ neg_ });
export const reciprocal = op({ reciprocal_ });
export const round = op({ round_ });
export const rsqrt = op({ rsqrt_ });
export const sigmoid = op({ sigmoid_ });
export const sign = op({ sign_ });
export const isNaN = op({ isNaN_ });
export const isInf = op({ isInf_ });
export const isFinite = op({ isFinite_ });
export const sin = op({ sin_ });
export const sinh = op({ sinh_ });
export const softplus = op({ softplus_ });
export const sqrt = op({ sqrt_ });
export const step = op({ step_ });
export const tan = op({ tan_ });
export const tanh = op({ tanh_ });
//# sourceMappingURL=unary_ops.js.map
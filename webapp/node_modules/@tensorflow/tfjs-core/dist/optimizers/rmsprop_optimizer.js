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
import { dispose, tidy } from '../globals';
import { zerosLike } from '../ops/ops';
import { registerClass } from '../serialization';
import { Optimizer } from './optimizer';
/** @doclink Optimizer */
export class RMSPropOptimizer extends Optimizer {
    constructor(learningRate, decay = 0.9, momentum = 0.0, epsilon = null, centered = false) {
        super();
        this.learningRate = learningRate;
        this.decay = decay;
        this.momentum = momentum;
        this.epsilon = epsilon;
        this.accumulatedMeanSquares = [];
        this.accumulatedMoments = [];
        this.accumulatedMeanGrads = [];
        this.centered = centered;
        if (epsilon == null) {
            this.epsilon = ENGINE.backend.epsilon();
        }
        if (learningRate == null) {
            throw new Error(`learningRate for RMSPropOptimizer must be defined.`);
        }
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            const trainable = false;
            if (this.accumulatedMeanSquares[i] == null) {
                this.accumulatedMeanSquares[i] = {
                    originalName: `${name}/rms`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            if (this.accumulatedMoments[i] == null) {
                this.accumulatedMoments[i] = {
                    originalName: `${name}/momentum`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            if (this.accumulatedMeanGrads[i] == null && this.centered) {
                this.accumulatedMeanGrads[i] = {
                    originalName: `${name}/mg`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const accumulatedMeanSquare = this.accumulatedMeanSquares[i].variable;
            const accumulatedMoments = this.accumulatedMoments[i].variable;
            tidy(() => {
                const newAccumulatedMeanSquare = accumulatedMeanSquare.mul(this.decay)
                    .add(gradient.square().mul(1 - this.decay));
                if (this.centered) {
                    const accumulatedMeanGrad = this.accumulatedMeanGrads[i].variable;
                    // Centered gradient
                    const newAccumulatedMeanGrad = accumulatedMeanGrad.mul(this.decay)
                        .add(gradient.mul(1 - this.decay));
                    const newAccumulatedMoments = accumulatedMoments.mul(this.momentum)
                        .add(gradient.mul(this.learningRate)
                        .div(newAccumulatedMeanSquare
                        .sub(newAccumulatedMeanGrad.square().add(this.epsilon))
                        .sqrt()));
                    accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
                    accumulatedMeanGrad.assign(newAccumulatedMeanGrad);
                    accumulatedMoments.assign(newAccumulatedMoments);
                    const newValue = value.sub(newAccumulatedMoments);
                    value.assign(newValue);
                }
                else {
                    // Plain gradient
                    const newAccumulatedMeanSquare = accumulatedMeanSquare.mul(this.decay)
                        .add(gradient.square().mul(1 - this.decay));
                    const newAccumulatedMoments = accumulatedMoments.mul(this.momentum)
                        .add(gradient.mul(this.learningRate)
                        .div(newAccumulatedMeanSquare.add(this.epsilon)
                        .sqrt()));
                    accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
                    accumulatedMoments.assign(newAccumulatedMoments);
                    const newValue = value.sub(newAccumulatedMoments);
                    value.assign(newValue);
                }
            });
        });
        this.incrementIterations();
    }
    dispose() {
        if (this.accumulatedMeanSquares != null) {
            dispose(this.accumulatedMeanSquares.map(v => v.variable));
        }
        if (this.accumulatedMeanGrads != null && this.centered) {
            dispose(this.accumulatedMeanGrads.map(v => v.variable));
        }
        if (this.accumulatedMoments != null) {
            dispose(this.accumulatedMoments.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        const variables = [...this.accumulatedMeanSquares, ...this.accumulatedMoments];
        if (this.centered) {
            variables.push(...this.accumulatedMeanGrads);
        }
        return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const variableCount = this.centered ? weightValues.length / 3 : weightValues.length / 2;
        const trainable = false;
        this.accumulatedMeanSquares =
            weightValues.slice(0, variableCount).map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        this.accumulatedMoments =
            weightValues.slice(variableCount, variableCount * 2)
                .map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        if (this.centered) {
            this.accumulatedMeanGrads =
                weightValues.slice(variableCount * 2, variableCount * 3)
                    .map(v => ({
                    originalName: v.name,
                    variable: v.tensor.variable(trainable)
                }));
        }
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'decay': this.decay,
            'momentum': this.momentum,
            'epsilon': this.epsilon,
            'centered': this.centered
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['decay'], config['momentum'], config['epsilon'], config['centered']);
    }
}
/** @nocollapse */
RMSPropOptimizer.className = 'RMSProp'; // Note: Name matters for Python compatibility.
registerClass(RMSPropOptimizer);
//# sourceMappingURL=rmsprop_optimizer.js.map
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { scalar } from '@tensorflow/tfjs-core';
import { TensorArray } from '../../executor/tensor_array';
import { getParamValue, getTensor } from './utils';
export const executeOp = async (node, tensorMap, context) => {
    switch (node.op) {
        case 'If':
        case 'StatelessIf': {
            const thenFunc = getParamValue('thenBranch', node, tensorMap, context);
            const elseFunc = getParamValue('elseBranch', node, tensorMap, context);
            const cond = getParamValue('cond', node, tensorMap, context);
            const args = getParamValue('args', node, tensorMap, context);
            const condValue = await cond.data();
            if (condValue[0]) {
                return context.functionMap[thenFunc].executeFunctionAsync(args);
            }
            else {
                return context.functionMap[elseFunc].executeFunctionAsync(args);
            }
        }
        case 'While':
        case 'StatelessWhile': {
            const bodyFunc = getParamValue('body', node, tensorMap, context);
            const condFunc = getParamValue('cond', node, tensorMap, context);
            const args = getParamValue('args', node, tensorMap, context);
            const condTensor = (await context.functionMap[condFunc].executeFunctionAsync(args))[0];
            let condValue = await condTensor.data();
            let result = args;
            while (condValue[0]) {
                result =
                    await context.functionMap[bodyFunc].executeFunctionAsync(result);
                const condTensor = (await context.functionMap[condFunc].executeFunctionAsync(result))[0];
                condValue = await condTensor.data();
            }
            return result;
        }
        case 'LoopCond':
            return [
                getParamValue('pred', node, tensorMap, context).clone()
            ];
        case 'Switch': {
            const pred = getParamValue('pred', node, tensorMap, context);
            const data = getParamValue('data', node, tensorMap, context);
            // Outputs nodes :0 => false, :1 => true
            return (await pred.data())[0] ? [undefined, data.clone()] :
                [data.clone(), undefined];
        }
        case 'Merge':
            const inputName = node.inputNames.find(name => getTensor(name, tensorMap, context) !== undefined);
            return inputName ? [getTensor(inputName, tensorMap, context).clone()] :
                undefined;
        case 'Enter':
            const frameId = getParamValue('frameName', node, tensorMap, context);
            const data = getParamValue('tensor', node, tensorMap, context);
            context.enterFrame(frameId);
            return [data.clone()];
        case 'Exit':
            const tensor = getParamValue('tensor', node, tensorMap, context);
            context.exitFrame();
            return [tensor.clone()];
        case 'NextIteration':
            const input = getParamValue('tensor', node, tensorMap, context);
            context.nextIteration();
            return [input.clone()];
        case 'TensorArrayV3':
            const size = getParamValue('size', node, tensorMap, context);
            const dtype = getParamValue('dtype', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const dynamicSize = getParamValue('dynamicSize', node, tensorMap, context);
            const clearAfterRead = getParamValue('clearAfterRead', node, tensorMap, context);
            const identicalElementShapes = getParamValue('identicalElementShapes', node, tensorMap, context);
            const name = getParamValue('name', node, tensorMap, context);
            const tensorArray = new TensorArray(name, dtype, size, elementShape, identicalElementShapes, dynamicSize, clearAfterRead);
            context.addTensorArray(tensorArray);
            return [scalar(tensorArray.id), scalar(1.0)];
        case 'TensorArrayWriteV3':
            const id = getParamValue('tensorArrayId', node, tensorMap, context);
            const index = getParamValue('index', node, tensorMap, context);
            const writeTensor = getParamValue('tensor', node, tensorMap, context);
            const writeTensorArray = context.getTensorArray(id);
            writeTensorArray.write(index, writeTensor);
            return [scalar(1.0)];
        case 'TensorArrayReadV3':
            const readId = getParamValue('tensorArrayId', node, tensorMap, context);
            const readIndex = getParamValue('index', node, tensorMap, context);
            const readTensorArray = context.getTensorArray(readId);
            return [readTensorArray.read(readIndex)];
        case 'TensorArrayGatherV3':
            const gatherId = getParamValue('tensorArrayId', node, tensorMap, context);
            const gatherIndices = getParamValue('indices', node, tensorMap, context);
            const gatherDtype = getParamValue('dtype', node, tensorMap, context);
            const gatherTensorArray = context.getTensorArray(gatherId);
            return [gatherTensorArray.gather(gatherIndices, gatherDtype)];
        case 'TensorArrayScatterV3':
            const scatterId = getParamValue('tensorArrayId', node, tensorMap, context);
            const scatterIndices = getParamValue('indices', node, tensorMap, context);
            const scatterTensor = getParamValue('tensor', node, tensorMap, context);
            const scatterTensorArray = context.getTensorArray(scatterId);
            scatterTensorArray.scatter(scatterIndices, scatterTensor);
            return [scalar(1.0)];
        case 'TensorArrayConcatV3':
            const concatId = getParamValue('tensorArrayId', node, tensorMap, context);
            const concatTensorArray = context.getTensorArray(concatId);
            const concatDtype = getParamValue('dtype', node, tensorMap, context);
            return [concatTensorArray.concat(concatDtype)];
        case 'TensorArraySplitV3':
            const splitId = getParamValue('tensorArrayId', node, tensorMap, context);
            const splitTensor = getParamValue('tensor', node, tensorMap, context);
            const lengths = getParamValue('lengths', node, tensorMap, context);
            const splitTensorArray = context.getTensorArray(splitId);
            splitTensorArray.split(lengths, splitTensor);
            return [scalar(1.0)];
        case 'TensorArraySizeV3':
            const sizeId = getParamValue('tensorArrayId', node, tensorMap, context);
            const sizeTensorArray = context.getTensorArray(sizeId);
            return [scalar(sizeTensorArray.size(), 'int32')];
        case 'TensorArrayCloseV3':
            const closeId = getParamValue('tensorArrayId', node, tensorMap, context);
            const closeTensorArray = context.getTensorArray(closeId);
            closeTensorArray.clearAndClose();
            return [scalar(0)];
        default:
            throw TypeError(`Node type ${node.op} is not implemented`);
    }
};
export const CATEGORY = 'control';
//# sourceMappingURL=control_executor.js.map
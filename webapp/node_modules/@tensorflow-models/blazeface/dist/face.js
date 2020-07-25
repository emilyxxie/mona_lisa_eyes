"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs-core");
const box_1 = require("./box");
const ANCHORS_CONFIG = {
    'strides': [8, 16],
    'anchors': [2, 6]
};
const NUM_LANDMARKS = 6;
function generateAnchors(width, height, outputSpec) {
    const anchors = [];
    for (let i = 0; i < outputSpec.strides.length; i++) {
        const stride = outputSpec.strides[i];
        const gridRows = Math.floor((height + stride - 1) / stride);
        const gridCols = Math.floor((width + stride - 1) / stride);
        const anchorsNum = outputSpec.anchors[i];
        for (let gridY = 0; gridY < gridRows; gridY++) {
            const anchorY = stride * (gridY + 0.5);
            for (let gridX = 0; gridX < gridCols; gridX++) {
                const anchorX = stride * (gridX + 0.5);
                for (let n = 0; n < anchorsNum; n++) {
                    anchors.push([anchorX, anchorY]);
                }
            }
        }
    }
    return anchors;
}
function decodeBounds(boxOutputs, anchors, inputSize) {
    const boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
    const centers = tf.add(boxStarts, anchors);
    const boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);
    const boxSizesNormalized = tf.div(boxSizes, inputSize);
    const centersNormalized = tf.div(centers, inputSize);
    const halfBoxSize = tf.div(boxSizesNormalized, 2);
    const starts = tf.sub(centersNormalized, halfBoxSize);
    const ends = tf.add(centersNormalized, halfBoxSize);
    const startNormalized = tf.mul(starts, inputSize);
    const endNormalized = tf.mul(ends, inputSize);
    const concatAxis = 1;
    return tf.concat2d([startNormalized, endNormalized], concatAxis);
}
function getInputTensorDimensions(input) {
    return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
        [input.height, input.width];
}
function flipFaceHorizontal(face, imageWidth) {
    let flippedTopLeft, flippedBottomRight, flippedLandmarks;
    if (face.topLeft instanceof tf.Tensor &&
        face.bottomRight instanceof tf.Tensor) {
        const [topLeft, bottomRight] = tf.tidy(() => {
            return [
                tf.concat([
                    tf.sub(imageWidth - 1, face.topLeft.slice(0, 1)),
                    face.topLeft.slice(1, 1)
                ]),
                tf.concat([
                    tf.sub(imageWidth - 1, face.bottomRight.slice(0, 1)),
                    face.bottomRight.slice(1, 1)
                ])
            ];
        });
        flippedTopLeft = topLeft;
        flippedBottomRight = bottomRight;
        if (face.landmarks != null) {
            flippedLandmarks = tf.tidy(() => {
                const a = tf.sub(tf.tensor1d([imageWidth - 1, 0]), face.landmarks);
                const b = tf.tensor1d([1, -1]);
                const product = tf.mul(a, b);
                return product;
            });
        }
    }
    else {
        const [topLeftX, topLeftY] = face.topLeft;
        const [bottomRightX, bottomRightY] = face.bottomRight;
        flippedTopLeft = [imageWidth - 1 - topLeftX, topLeftY];
        flippedBottomRight = [imageWidth - 1 - bottomRightX, bottomRightY];
        if (face.landmarks != null) {
            flippedLandmarks =
                face.landmarks.map((coord) => ([
                    imageWidth - 1 - coord[0],
                    coord[1]
                ]));
        }
    }
    const flippedFace = {
        topLeft: flippedTopLeft,
        bottomRight: flippedBottomRight
    };
    if (flippedLandmarks != null) {
        flippedFace.landmarks = flippedLandmarks;
    }
    if (face.probability != null) {
        flippedFace.probability = face.probability instanceof tf.Tensor ?
            face.probability.clone() :
            face.probability;
    }
    return flippedFace;
}
function scaleBoxFromPrediction(face, scaleFactor) {
    return tf.tidy(() => {
        let box;
        if (face.hasOwnProperty('box')) {
            box = face.box;
        }
        else {
            box = face;
        }
        return box_1.scaleBox(box, scaleFactor).startEndTensor.squeeze();
    });
}
class BlazeFaceModel {
    constructor(model, width, height, maxFaces, iouThreshold, scoreThreshold) {
        this.blazeFaceModel = model;
        this.width = width;
        this.height = height;
        this.maxFaces = maxFaces;
        this.anchorsData = generateAnchors(width, height, ANCHORS_CONFIG);
        this.anchors = tf.tensor2d(this.anchorsData);
        this.inputSizeData = [width, height];
        this.inputSize = tf.tensor1d([width, height]);
        this.iouThreshold = iouThreshold;
        this.scoreThreshold = scoreThreshold;
    }
    async getBoundingBoxes(inputImage, returnTensors, annotateBoxes = true) {
        const [detectedOutputs, boxes, scores] = tf.tidy(() => {
            const resizedImage = inputImage.resizeBilinear([this.width, this.height]);
            const normalizedImage = tf.mul(tf.sub(resizedImage.div(255), 0.5), 2);
            const batchedPrediction = this.blazeFaceModel.predict(normalizedImage);
            const prediction = batchedPrediction.squeeze();
            const decodedBounds = decodeBounds(prediction, this.anchors, this.inputSize);
            const logits = tf.slice(prediction, [0, 0], [-1, 1]);
            const scores = tf.sigmoid(logits).squeeze();
            return [prediction, decodedBounds, scores];
        });
        const savedConsoleWarnFn = console.warn;
        console.warn = () => { };
        const boxIndicesTensor = tf.image.nonMaxSuppression(boxes, scores, this.maxFaces, this.iouThreshold, this.scoreThreshold);
        console.warn = savedConsoleWarnFn;
        const boxIndices = await boxIndicesTensor.array();
        boxIndicesTensor.dispose();
        let boundingBoxes = boxIndices.map((boxIndex) => tf.slice(boxes, [boxIndex, 0], [1, -1]));
        if (!returnTensors) {
            boundingBoxes = await Promise.all(boundingBoxes.map(async (boundingBox) => {
                const vals = await boundingBox.array();
                boundingBox.dispose();
                return vals;
            }));
        }
        const originalHeight = inputImage.shape[1];
        const originalWidth = inputImage.shape[2];
        let scaleFactor;
        if (returnTensors) {
            scaleFactor = tf.div([originalWidth, originalHeight], this.inputSize);
        }
        else {
            scaleFactor = [
                originalWidth / this.inputSizeData[0],
                originalHeight / this.inputSizeData[1]
            ];
        }
        const annotatedBoxes = [];
        for (let i = 0; i < boundingBoxes.length; i++) {
            const boundingBox = boundingBoxes[i];
            const annotatedBox = tf.tidy(() => {
                const box = boundingBox instanceof tf.Tensor ?
                    box_1.createBox(boundingBox) :
                    box_1.createBox(tf.tensor2d(boundingBox));
                if (!annotateBoxes) {
                    return box;
                }
                const boxIndex = boxIndices[i];
                let anchor;
                if (returnTensors) {
                    anchor = this.anchors.slice([boxIndex, 0], [1, 2]);
                }
                else {
                    anchor = this.anchorsData[boxIndex];
                }
                const landmarks = tf.slice(detectedOutputs, [boxIndex, NUM_LANDMARKS - 1], [1, -1])
                    .squeeze()
                    .reshape([NUM_LANDMARKS, -1]);
                const probability = tf.slice(scores, [boxIndex], [1]);
                return { box, landmarks, probability, anchor };
            });
            annotatedBoxes.push(annotatedBox);
        }
        boxes.dispose();
        scores.dispose();
        detectedOutputs.dispose();
        return {
            boxes: annotatedBoxes,
            scaleFactor
        };
    }
    async estimateFaces(input, returnTensors = false, flipHorizontal = false, annotateBoxes = true) {
        const [, width] = getInputTensorDimensions(input);
        const image = tf.tidy(() => {
            if (!(input instanceof tf.Tensor)) {
                input = tf.browser.fromPixels(input);
            }
            return input.toFloat().expandDims(0);
        });
        const { boxes, scaleFactor } = await this.getBoundingBoxes(image, returnTensors, annotateBoxes);
        image.dispose();
        if (returnTensors) {
            return boxes.map((face) => {
                const scaledBox = scaleBoxFromPrediction(face, scaleFactor);
                let normalizedFace = {
                    topLeft: scaledBox.slice([0], [2]),
                    bottomRight: scaledBox.slice([2], [2])
                };
                if (annotateBoxes) {
                    const { landmarks, probability, anchor } = face;
                    const normalizedLandmarks = landmarks.add(anchor).mul(scaleFactor);
                    normalizedFace.landmarks = normalizedLandmarks;
                    normalizedFace.probability = probability;
                }
                if (flipHorizontal) {
                    normalizedFace = flipFaceHorizontal(normalizedFace, width);
                }
                return normalizedFace;
            });
        }
        return Promise.all(boxes.map(async (face) => {
            const scaledBox = scaleBoxFromPrediction(face, scaleFactor);
            let normalizedFace;
            if (!annotateBoxes) {
                const boxData = await scaledBox.array();
                normalizedFace = {
                    topLeft: boxData.slice(0, 2),
                    bottomRight: boxData.slice(2)
                };
            }
            else {
                const [landmarkData, boxData, probabilityData] = await Promise.all([face.landmarks, scaledBox, face.probability].map(async (d) => d.array()));
                const anchor = face.anchor;
                const [scaleFactorX, scaleFactorY] = scaleFactor;
                const scaledLandmarks = landmarkData
                    .map(landmark => ([
                    (landmark[0] + anchor[0]) * scaleFactorX,
                    (landmark[1] + anchor[1]) * scaleFactorY
                ]));
                normalizedFace = {
                    topLeft: boxData.slice(0, 2),
                    bottomRight: boxData.slice(2),
                    landmarks: scaledLandmarks,
                    probability: probabilityData
                };
                box_1.disposeBox(face.box);
                face.landmarks.dispose();
                face.probability.dispose();
            }
            scaledBox.dispose();
            if (flipHorizontal) {
                normalizedFace = flipFaceHorizontal(normalizedFace, width);
            }
            return normalizedFace;
        }));
    }
}
exports.BlazeFaceModel = BlazeFaceModel;
//# sourceMappingURL=face.js.map
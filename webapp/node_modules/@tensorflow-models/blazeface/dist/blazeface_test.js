"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs-core");
const jasmine_util_1 = require("@tensorflow/tfjs-core/dist/jasmine_util");
const blazeface = require("./index");
const test_util_1 = require("./test_util");
jasmine_util_1.describeWithFlags('BlazeFace', jasmine_util_1.NODE_ENVS, () => {
    let model;
    beforeAll(async () => {
        model = await blazeface.load();
    });
    it('estimateFaces does not leak memory', async () => {
        const input = tf.zeros([128, 128, 3]);
        const beforeTensors = tf.memory().numTensors;
        await model.estimateFaces(input);
        expect(tf.memory().numTensors).toEqual(beforeTensors);
    });
    it('estimateFaces returns objects with expected properties', async () => {
        const input = tf.tensor3d(test_util_1.stubbedImageVals, [128, 128, 3]);
        const result = await model.estimateFaces(input);
        const face = result[0];
        expect(face.topLeft).toBeDefined();
        expect(face.bottomRight).toBeDefined();
        expect(face.landmarks).toBeDefined();
        expect(face.probability).toBeDefined();
    });
});
//# sourceMappingURL=blazeface_test.js.map
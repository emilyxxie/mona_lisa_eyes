/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import "./style.css";
import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');


let model, ctx, videoWidth, videoHeight, video, canvas;

const state = {
  backend: 'wasm'
};

async function setupCamera() {
  video = document.getElementById('cam');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}


const map = (value, x1, y1, x2, y2) => (value - x1) * (y2 - x2) / (y1 - x1) + x2;

const oldImage = document.querySelector("#deepFakeImage");
let eye;

function moveEyes(leftEye, rightEye) {
    if (leftEye && rightEye) {
      eye = (leftEye + rightEye) / 2;

    } else if (leftEye < 0) {
      eye = rightEye;
    } else if (rightEye > video.width) {
      eye = leftEye
    }
    let headPos = Math.floor(map(eye, 0, video.width, 26, 55));
    headPos = Math.min(Math.max(headPos, 26), 54);
    if (!isNaN(headPos)) {
        const newImage = document.createElement("IMG");
        const ID = document.createElement("id");
        oldImage.src =  '../assets/frames/frame_' + headPos + '.png';
    }
}

const renderPrediction = async () => {

  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = true;
  const predictions = await model.estimateFaces(video, returnTensors, flipHorizontal, annotateBoxes);

  if (predictions.length > 0) {
    const landmarks = predictions[0].landmarks;
    const midEye = (landmarks[0][0] + landmarks[1][0]) / 2;
    moveEyes(landmarks[0][0], landmarks[1][0]);
  }

  setTimeout(function () {
    requestAnimationFrame(renderPrediction)
  }, 16);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;
  video.style.cssText = "-moz-transform: scale(-1, 1); \
    -webkit-transform: scale(-1, 1); -o-transform: scale(-1, 1); \
    transform: scale(-1, 1); filter: FlipH;";

  model = await blazeface.load();

  renderPrediction();
};

setupPage();



  // canvas = document.getElementById('output');
  // canvas.width = videoWidth;
  // canvas.height = videoHeight;
  // ctx = canvas.getContext('2d');
  // ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

// import "./style.css";
// import * as blazeface from '@tensorflow-models/blazeface';
// import * as tf from '@tensorflow/tfjs-core';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

// tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');

// let model, ctx, videoWidth, videoHeight, webCam, canvas;


// async function main() {
//   await tf.setBackend("wasm");
//   const webCam = document.getElementById("cam");
//   webCam.srcObject = await navigator.mediaDevices.getUserMedia({
//     "audio": false,
//     "video": { facingMode: "user" },
//   });

//   model = await blazeface.load();

//   renderPrediction();
// };




// const renderPrediction = async () => {
//   const returnTensors = false;
//   const flipHorizontal = true;
//   const annotateBoxes = true;
//   const predictions = await model.estimateFaces(webCam, returnTensors, flipHorizontal, annotateBoxes);

//   console.log("here are the predictions");
//   if (predictions.length > 0) {
//     console.log(predictions);
//     // ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // for (let i = 0; i < predictions.length; i++) {
//     //   const start = predictions[i].topLeft;
//     //   const end = predictions[i].bottomRight;
//     //   const size = [end[0] - start[0], end[1] - start[1]];
//     //   ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
//     //   ctx.fillRect(start[0], start[1], size[0], size[1]);

//     //   if (annotateBoxes) {
//     //     const landmarks = predictions[i].landmarks;

//     //     ctx.fillStyle = "blue";
//     //     for (let j = 0; j < landmarks.length; j++) {
//     //       const x = landmarks[j][0];
//     //       const y = landmarks[j][1];
//     //       ctx.fillRect(x, y, 5, 5);
//     //     }
//     //   }
//     // }
//   }
//   requestAnimationFrame(renderPrediction);
// };



// // async function main() {
// //     tf.setBackend('wasm').then(() => main());
// //     const model = await blazeface.load();
// //     console.log(model);
// // }



// WORKS FINE BELOW

// let cursorX, cursorY;


// const width = window.innerWidth;
// const height = window.innerHeight;

// document.onmousemove = function(e){
//     cursorX = e.pageX;
//     cursorY = e.pageY;
// }


// function checkCursor(){
//     const cursorPos = Math.floor(map(cursorX, 0, width, 26, 54));
//     if (!isNaN(cursorPos)) {
//         const oldImage = document.querySelector("#deepFakeImage");
//         const newImage = document.createElement("IMG");
//         const ID = document.createElement("id");
//         console.log("HELLO");
//         newImage.src =  '../assets/frames/frame_' + cursorPos + '.png';
//         newImage.id = "deepFakeImage";
//         oldImage.replaceWith(newImage);

//     }

// }


// setInterval(checkCursor, 10);

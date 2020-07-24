var cursorX;
var cursorY;

const width = window.innerWidth;
const height = window.innerHeight;

document.onmousemove = function(e){
    cursorX = e.pageX;
    cursorY = e.pageY;
}

setInterval(checkCursor, 100);



function checkCursor(){

    const cursorPos = Math.floor(map(cursorX, 0, width, 26, 54));
    if (!isNaN(cursorPos)) {

        // const toReplace = document.querySelector("#deepFake");
        const oldImage = document.querySelector("#deepFakeImage");
        const newImage = document.createElement("IMG");
        const ID = document.createElement("id");
        newImage.src =  '../frames/frame_' + cursorPos + '.png';
        newImage.id = "deepFakeImage";
        oldImage.replaceWith(newImage);

    }

}


const map = (value, x1, y1, x2, y2) => (value - x1) * (y2 - x2) / (y1 - x1) + x2;





async function setupCamera() {
    video = document.getElementById('video');

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


async function main() {
      video = document.getElementById('video');

      const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': { facingMode: 'user' },
      });
      video.srcObject = stream;

    // Load the model.
    const model = await blazeface.load();

    // Pass in an image or video to the model. The model returns an array of
    // bounding boxes, probabilities, and landmarks, one for each detected face.

    const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
    const predictions = await model.estimateFaces(document.querySelector("img"), returnTensors);

    if (predictions.length > 0) {
    /*
    `predictions` is an array of objects describing each detected face, for example:

    [
      {
        topLeft: [232.28, 145.26],
        bottomRight: [449.75, 308.36],
        probability: [0.998],
        landmarks: [
          [295.13, 177.64], // right eye
          [382.32, 175.56], // left eye
          [341.18, 205.03], // nose
          [345.12, 250.61], // mouth
          [252.76, 211.37], // right ear
          [431.20, 204.93] // left ear
        ]
      }
    ]
    */

    for (let i = 0; i < predictions.length; i++) {
      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];

      // Render a rectangle over each detected face.
      ctx.fillRect(start[0], start[1], size[0], size[1]);
    }
    }
}

main();
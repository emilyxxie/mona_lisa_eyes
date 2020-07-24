import "./style.css";

let cursorX;
let cursorY;

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
        newImage.src =  '../assets/frames/frame_' + cursorPos + '.png';
        newImage.id = "deepFakeImage";
        oldImage.replaceWith(newImage);

    }

}


const map = (value, x1, y1, x2, y2) => (value - x1) * (y2 - x2) / (y1 - x1) + x2;




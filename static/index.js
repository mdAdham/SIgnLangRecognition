const el = document.querySelector(".mouse");

function onMouseMove(e) {
  x = e.clientX;
  y = e.clientY;
  updateMouse(x, y);
}

function updateMouse(x, y) {
  el.style.transform = `translate(${x}px, ${y}px)`;
}

function mouseClick(x, y) {
  var ev = new MouseEvent("click", {
    view: window,
    bubbles: true,
    cancelable: true,
    screenX: x,
    screenY: y,
  });

  var el = document.elementFromPoint(x, y);

  el.dispatchEvent(ev);
  console.log("clicked at " + x + "," + y);
}

window.addEventListener("mousemove", onMouseMove);

function changeBG() {
  var x = Math.floor(Math.random() * 256);
  var y = Math.floor(Math.random() * 256);
  var z = Math.floor(Math.random() * 256);
  var bgColor = "rgb(" + x + "," + y + "," + z + ")";
  console.log(bgColor);

  document.body.style.background = bgColor;
}

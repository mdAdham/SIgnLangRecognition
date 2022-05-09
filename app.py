#Import necessary libraries
from flask import Flask, render_template, Response
from custom_ml_camera import gen_frames
import cv2
import jyserver.Flask as js
#Initialize the Flask app
app = Flask(__name__)

@js.use(app)
class App():
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.num_like = 0
        self.prev_gesture = None

    def log(self, msg):
        self.js.console.log(msg)

    def set_prev_gesture(self, gesture):
        self.prev_gesture = gesture
    
    def updateMousePos(self, percentage_x, percentage_y):
        # Update the mouse position on the browser, convert the landmark location to be relative to the browser
        browser_width = self.js.window.innerWidth
        browser_height = self.js.window.innerHeight
        x = int(percentage_x * int(browser_width))
        y = int(percentage_y * int(browser_height))
        self.js.document.getElementById("mouse").style.transform = "translate({x}px, {y}px)".format(x=x, y=y)
    
    def mouseClick(self, percentage_x, percentage_y):
        # Update the mouse position on the browser, convert the landmark location to be relative to the browser
        browser_width = self.js.window.innerWidth
        browser_height = self.js.window.innerHeight
        x = int(percentage_x * int(browser_width))
        y = int(percentage_y * int(browser_height))
        self.js.document.getElementById("mouse").display = "none"
        self.js.mouseClick(x, y)
        self.js.document.getElementById("mouse").display = "content"
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

@app.route('/')
def index():
     return App.render(render_template('index.html'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(App), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game2048')
def game2048():
    return App.render(render_template('game2048.html'))

if __name__ == "__main__":
    app.run(debug=True)
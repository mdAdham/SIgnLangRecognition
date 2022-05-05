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

    def log(self, msg):
        self.js.console.log(msg)
    
    def add_like(self):
        self.num_like += 1
        self.js.document.getElementById("num-likes").innerHTML = "Likes: " + str(self.num_like)
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

if __name__ == "__main__":
    app.run(debug=True)
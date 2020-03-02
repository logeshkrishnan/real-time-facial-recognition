from flask import Flask, request, jsonify, Response, json, render_template
import sqlite3, cv2, time, re, os, queue, threading
from flask_cors import CORS, cross_origin
from imutils.video import VideoStream
from datetime import datetime
import face_recognition
import numpy as np

# Initiating Flask App 
app = Flask(__name__)

# bufferless VideoCapture
class VideoCapture():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def get_frame(self):
        success, image = self.cap.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode(".jpg", image)
        return jpeg.tobytes()

# initialize the video stream and allow the camera sensor to warmup
vs = VideoCapture()
#time.sleep(2.0)

# Defining Routes
# Homepage
@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

# face recognition
def face_recog():

	known_face_encodings = []
	known_face_names = []
	known_faces_filenames = []

	for (dirpath, dirnames, filenames) in os.walk('assets/img/users/'):
	    known_faces_filenames.extend(filenames)
	    break
	    
	for filename in known_faces_filenames:
	    face = face_recognition.load_image_file('assets/img/users/' + filename)
	    known_face_names.append(re.sub("[0-9]",'', filename[:-4]))
	    known_face_encodings.append(face_recognition.face_encodings(face)[0])



	face_locations = []
	face_encodings = []
	face_names = []
	process_this_frame = True

	# grab global references to the video stream, output frame, and
	# lock variables
	while True:
		# Grab a single frame of video
		frame = vs.read()

		# Process every frame only one time
		if process_this_frame:
			# Find all faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(frame)
			face_encodings = face_recognition.face_encodings(frame, face_locations)

			# Initialize an array for the name of the detected faces
			face_names = []

			for face_encoding in face_encodings:
				# see if there is a match for the known faces
				matches = face_recognition.face_distance(known_face_encodings, face_encoding)
				name = "Unknown"

				# Or instead, use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:
					
					name = known_face_names[best_match_index]
					now = datetime.now()
					t_string = now.strftime("%H:%M:%S")
					d_string = now.strftime("%d-%m-%Y")

				face_names.append(name)

		process_this_frame = not process_this_frame

		# Dispplay this results
		for(top, right, bottom, left), name in zip(face_locations, face_names):

			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			# Draw a label with a face
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def generate(VideoCapture):
	while True:
		frame = VideoCapture.get_frame()
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(frame) + b'\r\n')

# route for Video feed
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(VideoCapture()),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	t = threading.Thread(target = face_recog, )
	t.daemon = True
	t.start()
	app.run(debug=True, threaded=True, use_reloader=False)

# Release the video streamer point
vs.release()

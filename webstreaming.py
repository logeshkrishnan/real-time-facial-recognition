# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
import face_recognition
from flask import render_template
import threading
import argparse
from datetime import datetime
import imutils
import time
import cv2
import os
import re
import numpy as np

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# Database Connection
db = os.path.join(os.getcwd(), 'database.db')
def database_connection():
	return sqlite3.connect(db, check_same_thread = False)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion():
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
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

	# loop over frames from the video stream
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

			# Initialisr json to export file
			json_to_export = {}
			# loop in every face detected
			for face_encoding in face_encodings:
				# see if there is a match for the known faces
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"

				# Or instead, use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:

					name = known_face_names[best_match_index]

					# Saving details to export
					json_to_export['name'] = name
					now = datetime.now()
					json_to_export['hour'] = now.strftime("%H:%M:%S")
					json_to_export['date'] = now.strftime("%d-%m-%Y")
					json_to_export['picture_array']  = frame.tolist()

					# * ---------- SEND data to API --------- *
					# Make a POST request to the API
					r = requests.post(url = "http://127.0.0.1:5000/receive_data", json = json_to_export)
					#print("Status :" r.status)

			# store the name in an array to display it later
			#face_names.append(name)
			# process every frame only one time
			process_this_frame = not process_this_frame

			# Dispplay this results
			for(top, right, bottom, left) in zip(face_locations):

				# Draw a box around the face
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

				# Draw a label with a face
				#font = cv2.FONT_HERSHEY_DUPLEX
				#cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

			# acquire the lock, set the output frame, and release the
			# lock
			with lock:
				outputFrame = frame.copy()

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion)
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()

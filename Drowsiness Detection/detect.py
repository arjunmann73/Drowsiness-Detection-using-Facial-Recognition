from threading import Thread
import time
import numpy as np
import functions
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="Enter the path to the shape predictor.")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="Enter the path to the alarm file.")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="Webcam (can change it to an external webcam)")
args = vars(ap.parse_args())


threshold = 0.3
threshold_frames = 50


count = 0
Al_on = False


print("Finding facial predictor!")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("Starting Video Stream!")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)


while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)


		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = functions.eye_aspect_ratio(leftEye)
		rightEAR = functions.eye_aspect_ratio(rightEye)


		ear = (leftEAR + rightEAR) / 2.0


		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


		if ear < threshold:
			count += 1


			if count >= threshold_frames:

				if not Al_on:
					Al_on = True


					if args["alarm"] != "":
						t = Thread(target=functions.sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()


				cv2.putText(frame, "WAKE UP!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		else:
			count = 0
			Al_on = False


		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("e"):
		break

cv2.destroyAllWindows()
vs.stop()

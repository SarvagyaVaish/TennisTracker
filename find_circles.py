# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
scale = 0.5

target_center = [0, 0]
target_history = []
TARGET_SMOOTHING_WINDOW = 5
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	#
	# Process the image to find the a circle
	#
	
	# Resize
	orig_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

	# Grayscale
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

	# Smooth
	image = cv2.GaussianBlur(image, (5, 5), 0)
	image = cv2.medianBlur(image, 5)

	# Hough circles
	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
								param1=50, param2=60, minRadius=0, maxRadius=0)

	# Find average center of all detected circles
	current_target_center = target_center
	if circles != None:
		current_target_center[0] = 0.
		current_target_center[1] = 0.
		for i in circles[0, :]:
			current_target_center[0] += i[0]
			current_target_center[1] += i[1]
			# cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
			# cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

		current_target_center[0] /= len(circles[0, :])
		current_target_center[1] /= len(circles[0, :])

	# Smooth target location
	target_history.append(current_target_center)
	while len(target_history) > TARGET_SMOOTHING_WINDOW:
		target_history.pop(0)

	target_center = np.mean(np.array(target_history), 0)

	cv2.circle(orig_image, (int(target_center[0]), int(target_center[1])), 2, (0, 0, 255), 3)

	# show the frame
	cv2.imshow("Frame", orig_image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


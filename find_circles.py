from __future__ import division
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import Adafruit_PCA9685


TARGET_SMOOTHING_WINDOW = 3
SCALE = 0.5


# Servo setup
pwm = Adafruit_PCA9685.PCA9685()
servo_min = 150  # Min pulse length out of 4096
servo_max = 600  # Max pulse length out of 4096
pwm.set_pwm_freq(60) # Set frequency to 60hz, good for servos.


def set_servo_position(position):
    pwm.set_pwm(14, 0, int(position))


def find_target(image):
    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.medianBlur(image, 5)

    # Hough circles
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=60, minRadius=0, maxRadius=0)

    # Find average center of all detected circles
    current_target_center = [-1, -1]
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

    return current_target_center


if __name__ == '__main__':
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # target location
    target_center = [0, 0]
    target_history = []
    no_target_count = 0
    sweep_mode = True
    sweep_direction = 1
    sweep_speed = 10

    # servo target | PID params
    servo_target = (servo_min * 2. + servo_max) / 3.
    servo_target = servo_min
    pid_p = 0.2
    pid_d = 0.0
    pid_i = 0.0
    cte = 0.
    cte_int = 0.

    set_servo_position(servo_target)

    # allow the camera to warmup
    time.sleep(1.)

    # capture frames from the camera
    iter_count = 0
    t1 = time.time()
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        #
        # Grab the raw NumPy array representing the image
        #
        orig_image = frame.array
        orig_image = cv2.resize(orig_image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        rows, cols, _ = np.shape(orig_image)

        #
        # Process the image to find target
        #
        new_target = find_target(orig_image)

        # Use last known location if target not found
        if new_target[0] == -1:
            new_target = list(target_center)
            no_target_count += 1
            # Go into sweep mode if we are missing target for a while
            if no_target_count > 5:
                sweep_mode = True
        else:
            no_target_count = 0
            sweep_mode = False

        # Sweep
        if sweep_mode:
            servo_target += sweep_direction * sweep_speed
            if servo_target < servo_min or servo_target > servo_max:
                sweep_direction *= -1

        # Use PID to center target
        else:
            # Smooth target location
            target_history.append(new_target)
            while len(target_history) > TARGET_SMOOTHING_WINDOW:
                target_history.pop(0)

            target_center = np.mean(np.array(target_history), 0)
            
            curr_cte = target_center[0] - cols / 2.
            cte_diff = curr_cte - cte
            cte = curr_cte
            cte_int += curr_cte

            servo_target -= (pid_p * cte  +  pid_d * cte_diff  +  pid_i * cte_int)
            servo_target = min(servo_target, servo_max)
            servo_target = max(servo_target, servo_min)

        #
        # Control Servo
        #
        set_servo_position(servo_target)

        #
        # Visualize
        #
        cv2.circle(orig_image, (int(target_center[0]), int(target_center[1])), 2, (0, 0, 255), 3)
        cv2.imshow("Frame", orig_image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        iter_count += 1

    t2 = time.time()
    print (t2 - t1) / iter_count


# import cv2 to capture videofeed
import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mount_everest.jpg')
mountain = cv2.resize(mountain, (640, 480))  # Resize the mountain image to match the video feed

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # perform bitwise and operation to extract foreground/person
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # combine the result with the mountain image
        output = cv2.addWeighted(result, 1, mountain, 0.5, 0)

        # show the resulting image
        cv2.imshow('frame', output)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()

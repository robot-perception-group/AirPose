import numpy as np
import cv2
from cv2 import aruco
import os

cap = cv2.VideoCapture(0)

img_counter = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = os.path.join("/is/ps3/nsaini/projects/copenet_real_data/webcam_chess_images","opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
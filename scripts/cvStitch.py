import cv2
import numpy as np

def load(sourcePath):
    camera = cv2.VideoCapture(str(sourcePath))
    frames = []
    while True:
        successReturn, currFrame = camera.read()
        if not successReturn or currFrame is None:
            break
        frames.append(currFrame)
    return frames


stitchy = cv2.Stitcher.create()
(dummy, output) = stitchy.stitch(load("/Users/joshchung/Desktop/IMG_6637480.MOV"))

if dummy != cv2.STITCHER_OK:
    # checking if the stitching procedure is successful
    # .stitch() function returns a true value if stitching is
    # done successfully
    print("stitching ain't successful")
else:
    print('Your Panorama is ready!!!')

# final output
cv2.imwrite('finalresult.png', output)

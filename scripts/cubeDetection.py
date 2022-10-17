import cv2
import numpy as np

img = cv2.imread('/Users/joshchung/Desktop/thesis/imageStitch/scripts/blackcubesStitched15.png')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thrash = cv2.threshold(imgGry, 60, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

thresholdMin = 500
thresholdMax = 2000
temp = []
for cou in contours:
    if cv2.contourArea(cou)>=thresholdMin and cv2.contourArea(cou)<=thresholdMax:
        temp.append(cou)


cv2.drawContours(img, temp, -1, (0,255,0), 3)
cv2.imwrite("blobblackcubedetection.png", img)



"""for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (255, 0, 0), 5)"""

#cv2.imwrite('blackcubedetection.png', img)
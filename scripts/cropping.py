import cv2
import numpy as np

def checkLeft(img):
    for i in range(len(img)):
        if(img[i][0]== 0):
            return True
    return False
def checkTop(img):
    for i in range(len(img[0])):
        if(img[0][i]== 0):
            return True
    return False
def checkRight(img):
    for i in range(len(img)):
        if(img[i][-1]== 0):
            return True
    return False
def checkDown(img):
    for i in range(len(img)):
        if(img[-1][i]== 0):
            return True
    return False

def crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)
    while True:
        state = 0
        if(checkTop(thresh)):
            thresh = np.delete(thresh, 0, axis=0)
            img = np.delete(img, 0, axis=0)
            state +=1
        if (checkDown(thresh)):
            thresh = np.delete(thresh, len(thresh)-1, axis=0)
            img = np.delete(img, len(img) - 1, axis=0)
            state += 1
        if (checkLeft(thresh)):
            thresh = np.delete(thresh, 0, axis=1)
            img = np.delete(img, 0, axis=1)
            state += 1
        if (checkRight(thresh)):
            thresh = np.delete(thresh, len(img[0])-1, axis=1)
            img = np.delete(img, len(img[0]) - 1, axis=1)
            state += 1
        if(state==0):
            return img



img = cv2.imread('1.png')

cv2.imwrite('0croppedmaybe.png',crop(img))
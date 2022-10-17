import cv2
import numpy as np
import threading
import queue

def load(sourcePath):
    camera = cv2.VideoCapture(str(sourcePath))
    frames = []
    while True:
        successReturn, currFrame = camera.read()
        if not successReturn or currFrame is None:
            break
        frames.append(currFrame)
    return frames

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
        if(img[i][len(img[i])-1]== 0):
            return True
    return False
def checkDown(img):
    for i in range(len(img[0])):
        if(img[len(img)-1][i]== 0):
            return True
    return False

def crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
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

def stitcherThread(threadNumber, frameList, numberOfThreads, q):
    # thread number
    # frames: list of frames
    # number of threads
    # q for getting return values
    if(threadNumber>=numberOfThreads):
        exit("Error with threadNumber to numberOfThreads")
    stitchAmount = int(len(frameList) / numberOfThreads)
    start = stitchAmount * threadNumber
    end = stitchAmount * threadNumber + stitchAmount - 1
    stitcher = ImageStitcher()
    #print("start "+str(start))
    #rint("end " + str(end))
    if threadNumber == numberOfThreads - 1:
        for i in range(start, len(frameList)):
            #print("adding"+str(i))
            stitcher.add_image(frameList[i])
    else:
        for i in range(start, end):
            #print("adding" + str(i))
            stitcher.add_image(frameList[i])

    q.put((threadNumber, crop(stitcher.image())))

class ImageStitcher:
    def __init__(self, min_num: int = 10, lowe: float = 0.7, knn_clusters: int = 2):
        self.min_num = min_num
        self.lowe = lowe
        self.knn_clusters = knn_clusters
        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
        self.sift = cv2.SIFT_create()
        self.result_image = None
        self.result_image_gray = None

    def add_image(self, image: np.ndarray):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        temp = self.result_image
        if self.result_image is None:
            self.result_image = image
            self.result_image_gray = image_gray
            return
        result_features = self.sift.detectAndCompute(self.result_image_gray, None)
        image_features = self.sift.detectAndCompute(image_gray, None)
        matches_src, matches_dst, n_matches = self.compute_matches(result_features, image_features, matcher=self.flann,
                                                                   knn=self.knn_clusters, lowe=self.lowe)
        if n_matches < self.min_num:
            print("not enough matches")
            return #("ERROR: Not enough features")
        homography, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
        self.result_image = self.combine_images(image, self.result_image, homography)
        if(np.size(self.result_image, 0) == np.size(image, 0) and np.size(self.result_image, 1) == np.size(image, 1)):
            self.result_image = temp
        else:
            self.result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2GRAY)

    def compute_matches(self, features0, features1, matcher, knn=5, lowe=0.7): #knn=5, lowe=0.7
        keypoints0, descriptors0 = features0
        keypoints1, descriptors1 = features1
        matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)
        positive = []
        for match0, match1 in matches:
            if match0.distance < lowe * match1.distance:
                positive.append(match0)
        src_pts = np.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=np.float32)
        src_pts = src_pts.reshape((-1, 1, 2))
        dst_pts = np.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=np.float32)
        dst_pts = dst_pts.reshape((-1, 1, 2))
        return src_pts, dst_pts, len(positive)

    def combine_images(self, img0, img1, h_matrix):
        points0 = np.array([[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]],
                              dtype=np.float32)
        points0 = points0.reshape((-1, 1, 2))
        points1 = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]],
                              dtype=np.float32)
        points1 = points1.reshape((-1, 1, 2))
        points2 = cv2.perspectiveTransform(points1, h_matrix)
        points = np.concatenate((points0, points2), axis=0)
        [x_min, y_min] = (points.min(axis=0).ravel() - .1).astype(np.int32)
        [x_max, y_max] = (points.max(axis=0).ravel() + .1).astype(np.int32)
        h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
        output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
        return output_img

    def image(self):
        return self.result_image

def helper(sourcePath, numberOfThreads, nameToWrite):
    frames = load(sourcePath)
    q = queue.Queue()
    threads = []
    for i in range(numberOfThreads):
        t = threading.Thread(target=stitcherThread, args=(i, frames, numberOfThreads, q,))
        threads.append(t)
    for i in range(numberOfThreads):
        threads[i].start()
    for i in range(numberOfThreads):
        threads[i].join()
        #print("joining" + str(i))
    rest = []
    while (q.empty() == False):
        item = q.get()
        rest.append(item)
    rest.sort(key=lambda x: x[0])
    stitcher = ImageStitcher()
    for i in range(len(rest)):
        #cv2.imwrite(str(i) + ".png", rest[i][1])
        stitcher.add_image(rest[i][1])
        #cv2.imwrite(str(i) + "fromstitcher.png", stitcher.image())

    cv2.imwrite(nameToWrite, stitcher.image())


sourcePath = "/Users/joshchung/Desktop/videos/blackcubes480.MOV"
numberOfThreads = 15
helper(sourcePath, numberOfThreads, "blackcubesStitched15.png")


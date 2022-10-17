import cv2
import numpy

def load(sourcePath):
    camera = cv2.VideoCapture(str(sourcePath))
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            break
        yield frame

def stitchVideo(sourcePath):
    stitcher = ImageStitcher()
    counter = 0
    for idx, frame in enumerate(load(sourcePath)):
        print(counter)
        counter += 1
        stitcher.add_image(frame)
    cv2.imwrite("tempor.png", stitcher.image())

class ImageStitcher:
    def __init__(self, min_num: int = 10, lowe: float = 0.7, knn_clusters: int = 2):
        self.min_num = min_num
        self.lowe = lowe
        self.knn_clusters = knn_clusters
        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
        self.sift = cv2.SIFT_create()
        self.result_image = None
        self.result_image_gray = None

    def add_image(self, image: numpy.ndarray):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.result_image is None:
            self.result_image = image
            self.result_image_gray = image_gray
            return
        result_features = self.sift.detectAndCompute(self.result_image_gray, None)
        image_features = self.sift.detectAndCompute(image_gray, None)
        matches_src, matches_dst, n_matches = self.compute_matches(result_features, image_features, matcher=self.flann, knn=self.knn_clusters, lowe=self.lowe)
        if n_matches < self.min_num:
            print("ERROR: Not enough features")
            return
        homography, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
        self.result_image = self.combine_images(image, self.result_image, homography)
        self.result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2GRAY)

    def compute_matches(self, features0, features1, matcher, knn=5, lowe=0.7):
        keypoints0, descriptors0 = features0
        keypoints1, descriptors1 = features1
        matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)
        positive = []
        for match0, match1 in matches:
            if match0.distance < lowe * match1.distance:
                positive.append(match0)
        src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=numpy.float32)
        src_pts = src_pts.reshape((-1, 1, 2))
        dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=numpy.float32)
        dst_pts = dst_pts.reshape((-1, 1, 2))
        return src_pts, dst_pts, len(positive)

    def combine_images(self, img0, img1, h_matrix):
        points0 = numpy.array([[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=numpy.float32)
        points0 = points0.reshape((-1, 1, 2))
        points1 = numpy.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=numpy.float32)
        points1 = points1.reshape((-1, 1, 2))
        points2 = cv2.perspectiveTransform(points1, h_matrix)
        points = numpy.concatenate((points0, points2), axis=0)
        [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(numpy.int32)
        [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(numpy.int32)
        h_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
        output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
        return output_img

    def image(self):
        return self.result_image

sourcePath = "/Users/joshchung/Desktop/videos/IMG_6637480.MOV"
stitchVideo(sourcePath)
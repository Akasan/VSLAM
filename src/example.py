from camera import VideoCapture
from feature_extractor import FeatureExtractor


cap = VideoCapture("../sample/test_countryroad_reverse.mp4")
fe = FeatureExtractor()
for i in range(1000):
    _, frame = cap.read()
    shape = frame.shape
    fe.draw_keypoints(frame, divide=(5, 5))

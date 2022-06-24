from typing import Tuple
import numpy as np
import cv2
from enum import Enum

class FeatureType(Enum):
    ORB = 1


class FeatureExtractor:
    def __init__(self, feature_type: FeatureType = FeatureType.ORB):
        self.FEATURE_TYPE = feature_type
        self._prepare_feature_extractor()

    def _prepare_feature_extractor(self):
        if self.FEATURE_TYPE == FeatureType.ORB:
            self._feature_extractor = cv2.ORB_create()

    def extract_keypoints(self, frame: np.ndarray, divide: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, np.ndarray]:
        shape = frame.shape
        kp = []

        for dh in range(divide[0]):
            for dw in range(divide[1]):
                sub_frame = frame[shape[0]*dh//divide[0]: shape[0]*(dh+1)//divide[0],
                                 shape[1]*dw//divide[1]: shape[1]*(dw+1)//divide[1], :]
                _kp = self._feature_extractor.detect(sub_frame)

                for i in range(len(_kp)):
                    _kp[i].pt = (_kp[i].pt[0]+shape[1]*dw//divide[1], _kp[i].pt[1]+shape[0]*dh//divide[0])

                kp += _kp

        return kp

    def draw_keypoints(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), divide: Tuple[int, int] = (1, 1)):
        kp = self.extract_keypoints(frame, divide)
        kp_frame = cv2.drawKeypoints(frame, kp, None, color, flags=0)

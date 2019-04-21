import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from ..utillibs import Util

"""
特徴抽出クラス
"""
class Feature(object):
    def __init__(self, img):
        self.img = img
        pass

    def calcORB(self, size=None):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        detector = cv2.ORB_create()
        (kp, des) = detector.detectAndCompute(img, None)
        return des

    def calcAKAZE(self, size=None):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        detector = cv2.AKAZE_create()
        (kp, des) = detector.detectAndCompute(img, None)
        return des

    def calcHist(self, size=None):
        if size != None:
            img = cv2.resize(self.img, size)
        hist = cv2.calcHist([img],[0], None, [256], [0, 256])
        return hist

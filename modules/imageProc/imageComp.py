import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from ..utillibs import Util

class ImageComp(object):
    ARG_INDEX = ['target', 'source', 'histCorr', 'histBhat', 'AKAZE', 'ORB']
    def __init__(self, INPUT_DIR, OUTPUT_DIR):
        self.INPUT_DIR = INPUT_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
    def execCompAll(self):
        df = pd.DataFrame(columns=ImageComp.ARG_INDEX)
        files = os.listdir(self.INPUT_DIR)
        for file in files:
            if file == '.DS_Store':
                continue
            dfTmp = self.__comp(file)
            df = pd.concat([df,dfTmp],axis=0)
        # save
        df.to_csv(self.OUTPUT_DIR + "comp_" + "All" + "_" + Util.getNow() + ".csv")
        return df

    def execComp(self, TARGET_FILE=None):
        df = self.__comp(TARGET_FILE)
        # save
        df.to_csv(self.OUTPUT_DIR + "comp_" + TARGET_FILE + "_" + Util.getNow() + ".csv")
        return df

    def __comp(self, TARGET_FILE=None):
        # Target
        valT = self.__calcVal(TARGET_FILE)
        # Comp
        df = pd.DataFrame(columns=ImageComp.ARG_INDEX)
        files = os.listdir(self.INPUT_DIR)
        for file in files:
            if file == '.DS_Store' or file == TARGET_FILE:
                continue
            valS = self.__calcVal(file)
            cVal = self.__compVal(valT,valS)
            df = df.append(cVal, ignore_index=True)
        return df

    def __calcVal(self, TARGET_FILE):
        val = {}
        print(self.INPUT_DIR + TARGET_FILE)
        img = cv2.imread(self.INPUT_DIR + TARGET_FILE)
        # info
        val['name'] = TARGET_FILE
        # calc(Histgram)
        val['hist'] = self.calcHist(img,None)
        # calc(AKAZE,ORB)
        val['AKAZE'] = self.calcAKAZE(img,None)
        val['ORB'] = self.calcORB(img,None)
        return val

    def __compVal(self, valT,valS):
        data = []
        data.append(valT['name'])
        data.append(valS['name'])
        # comp(Histgram-Corr,Bhattacharyya)
        data.append(cv2.compareHist(valT['hist'], valS['hist'], 0))
        data.append(cv2.compareHist(valT['hist'], valS['hist'], 3))
        # comp(BFM-AKAZE,ORB)
        data.append(self.compBFM(valT['AKAZE'], valS['AKAZE']))
        data.append(self.compBFM(valT['ORB'], valS['ORB']))
        # out
        return pd.Series(data=data, index=ImageComp.ARG_INDEX)

    ##-----------------------
    ## CalcFeature
    ##-----------------------
    def calcHist(self, img, IMG_SIZE=(200,200)):
        if IMG_SIZE != None:
            img = cv2.resize(img, IMG_SIZE)
        hist = cv2.calcHist([img],[0], None, [256], [0, 256])
        return hist

    def calcAKAZE(self, img, IMG_SIZE=(200,200)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        detector = cv2.AKAZE_create()
        (kp, des) = detector.detectAndCompute(img, None)
        return des

    def calcORB(self, img, IMG_SIZE=(200,200)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        detector = cv2.ORB_create()
        (kp, des) = detector.detectAndCompute(img, None)
        return des

    ##-----------------------
    ## CompFeature
    ##-----------------------
    def compBFM(self, desT, desS):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(desT, desS)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        return ret

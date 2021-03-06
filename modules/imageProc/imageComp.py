import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import pathlib
# from pdf2image import convert_from_path, convert_from_bytes
from ..utillibs import Util


class ImageComp(object):
    ARG_INDEX = ['target', 'source', 'histCorr', 'histBhat', 'AKAZE', 'ORB']
    ARGS = ['histCorr', 'histBhat', 'AKAZE', 'ORB']
    def __init__(self, INPUT_BASE, OUTPUT_BASE, FEATURE_BASE):
        self.INPUT_BASE = INPUT_BASE
        self.OUTPUT_BASE = OUTPUT_BASE
        self.FEATURE_BASE = FEATURE_BASE

    def __initComp(self, args=None):
        self.NO = Util.getNow()
        pOutput = (pathlib.Path(self.OUTPUT_BASE) / self.NO)
        pFeature = (pathlib.Path(self.FEATURE_BASE) / self.NO)
        pOutput.mkdir(exist_ok=True)
        pFeature.mkdir(exist_ok=True)
        self.OUTPUT_DIR = str(pOutput) + "/"
        self.FEATURE_DIR = str(pFeature) + "/"
        # 処理対象のアルゴリズム
        self.ARGS = ImageComp.ARGS
        if args != None self.ARGS = args

    def execComp(self, feature_dir=None, arg=None, target):
        self.__initComp(arg)
        files = os.listdir(self.INPUT_BASE)

        # calc source feature
        for file in files:
            if file == '.DS_Store':
                continue
            # calc Feature
            val = self.__calcFeature(file)
            # save Feature
            self.__saveVal(val)

        img = cv2.imread(self.INPUT_BASE + TARGET_FILE)
        if arg == "histCorr":
            # calc feature
            # save feature
            # comp feature
            pass
        elif arg == "histBhat":
            pass

    # 特徴量の比較
    def __comp(self, kind):
        print(self.FEATURE_DIR + "feature_" + kind + ".csv")
        dfFeature = pd.read_csv(self.FEATURE_DIR + "feature_" + kind + ".csv")
        if kind == "hist":
            self.__compHists(dfFeature)


    # def __comp(self, dfVal, OUTFILE, TARGET_FILE=None):
    #     dfComp = pd.DataFrame(columns=ImageComp.ARG_INDEX)
    #     print(dfVal)
    #     for i, irow in dfVal.iterrows():
    #         if TARGET_FILE != None and irow['name'] != TARGET_FILE:
    #             continue
    #         for j, jrow in dfVal.iterrows():
    #             if i==j:
    #                 continue
    #             psComp = self.__compVal(irow, jrow)
    #             psComp.to_csv(self.OUTPUT_DIR + OUTFILE, mode='a')
    #             # dfComp.append(psComp, ignore_index=True)

    def __compHists(self, dfHist, TARGET_FILE=None):
        for i, irow in dfHist.iterrows():
            if TARGET_FILE != None and irow['name'] != TARGET_FILE:
                continue
            for j, jrow in dfHist.iterrows():
                if i==j:
                    continue
                print("comp start")
                tmp = self.__loadFeature(irow)
                print(type(tmp),tmp.shape)
                compHistCorr = self.__compHist(self.__loadFeature(irow),self.__loadFeature(jrow), 0)
                compHistBhat = self.__compHist(irow[2:].values, jrow[2:].values, 3)
                print("histCorr:" + compHistCorr)
                print("histBhat:" + compHistBhat)
                # psComp = self.__compVal(irow, jrow)
                # psComp.to_csv(self.OUTPUT_DIR + OUTFILE, mode='a')

    def __compHist(self, ndHistT, ndHistS, method):
        return cv2.compareHist(H1=ndHistT, H2=ndHistS, method=method)


    # # 特徴量の比較
    # def __comp(self, TARGET_FILE=None):
    #     # Target
    #     valT = self.__calcVal(TARGET_FILE)
    #     # Comp
    #     df = pd.DataFrame(columns=ImageComp.ARG_INDEX)
    #     files = os.listdir(self.INPUT_DIR)
    #     for file in files:
    #         if file == '.DS_Store' or file == TARGET_FILE:
    #             continue
    #         valS = self.__calcVal(file)
    #         cVal = self.__compVal(valT,valS)
    #         df = df.append(cVal, ignore_index=True)
    #     return df

    # 特徴量の保存
    def __saveVal(self, listVal):
        # OUTPUT
        FEATURE_HIST = self.FEATURE_DIR + "feature_" + "hist" + ".csv"
        FEATURE_AKAZE = self.FEATURE_DIR + "feature_" + "AKAZE" + ".csv"
        FEATURE_ORB = self.FEATURE_DIR + "feature_" + "ORB" + ".csv"
        # Save(hist)
        df = pd.DataFrame(listVal['hist'].T)
        df_hist = pd.concat([pd.DataFrame([[listVal['name']]]*len(df)),df], axis=1)
        df_hist.to_csv(FEATURE_HIST, mode="a", header=False)
        # Save(AKAZE)
        df = pd.DataFrame(listVal['AKAZE'])
        df_AKAZE = pd.concat([pd.DataFrame([[listVal['name']]]*len(df)),df], axis=1)
        df_AKAZE.to_csv(FEATURE_AKAZE, mode="a", header=False, index=False)
        # Save(ORB)
        df = pd.DataFrame(listVal['ORB'])
        df_ORB = pd.concat([pd.DataFrame([[listVal['name']]]*len(df)),df], axis=1)
        df_ORB.to_csv(FEATURE_ORB, mode="a", header=False, index=False)

    # 特徴量のロード
    def __loadVal(self, FEATURE_FILE):
        return pd.read_csv(OUTPUT_DIR + FEATURE_FILE)

    def __loadFeature(self, ps):
        return ps[2:].values.reshape([-1,1])


    # 特徴量の計算
    def __calcFeature(self, TARGET_FILE):
        val = {}
        print(self.INPUT_BASE + TARGET_FILE)
        img = cv2.imread(self.INPUT_BASE + TARGET_FILE)
        # info
        val['name'] = TARGET_FILE
        # calc(Histgram)
        val['hist'] = self.calcHist(img,None)
        # calc(AKAZE,ORB)
        val['AKAZE'] = self.calcAKAZE(img,None)
        val['ORB'] = self.calcORB(img,None)
        return val

    # 特徴量の比較
    def __compVal(self, psValT, psValS):
        ps = pd.Series(index=ImageComp.ARG_INDEX)
        ps['target'] = psValT['name']
        ps['source'] = psValS['name']
        # 'histCorr', 'histBhat', 'AKAZE', 'ORB'
        # comp(Histgram-Corr,Bhattacharyya)
        ps['histCorr'] = cv2.compareHist(valT['hist'], valS['hist'], 0)
        ps['histBhat'] = cv2.compareHist(valT['hist'], valS['hist'], 3)
        # comp(BFM-AKAZE,ORB)
        ps['AKAZE'] = self.compBFM(valT['AKAZE'], valS['AKAZE'])
        ps['ORB'] = self.compBFM(valT['ORB'], valS['ORB'])
        # out
        return ps

    # # 特徴量の比較
    # def __compVal(self, valT,valS):
    #     data = []
    #     data.append(valT['name'])
    #     data.append(valS['name'])
    #     # comp(Histgram-Corr,Bhattacharyya)
    #     data.append(cv2.compareHist(valT['hist'], valS['hist'], 0))
    #     data.append(cv2.compareHist(valT['hist'], valS['hist'], 3))
    #     # comp(BFM-AKAZE,ORB)
    #     data.append(self.compBFM(valT['AKAZE'], valS['AKAZE']))
    #     data.append(self.compBFM(valT['ORB'], valS['ORB']))
    #     # out
    #     return pd.Series(data=data, index=ImageComp.ARG_INDEX)

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
        print(type(des),des.shape)
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

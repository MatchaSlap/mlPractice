import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
# from pdf2image import convert_from_path, convert_from_bytes
from ..utillibs import Util


class ImageComp(object):
    ARG_INDEX = ['target', 'source', 'histCorr', 'histBhat', 'AKAZE', 'ORB']
    def __init__(self, INPUT_DIR, OUTPUT_DIR):
        self.INPUT_DIR = INPUT_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
    def execCompAll(self, FEATURE_DIR=None):
        self.NO = Util.getNow()
        outFile = "comp_All_" + self.NO + ".csv"

        dfVal = pd.DataFrame(columns=['name','hist','AKAZE','ORB'])
        # CalcFeature
        if FEATURE_DIR == None:
            files = os.listdir(self.INPUT_DIR)
            for file in files:
                if file == '.DS_Store':
                    continue
                val = self.__calcVal(file)
                # dfVal.append(val, ignore_index=True)
                # save Feature
                self.__saveVal(val)
        else:
            # 読み込み処理をあとで書く
            dfVal = __loadVal(FEATURE_FILE)

        # CompareFeature
        # self.__comp(dfVal=dfVal, OUTFILE=outFile)

        # (self.__comp(dfVal)).to_csv(self.OUTPUT_DIR + outFile, mode='a')
        # dfComp.to_csv(self.OUTPUT_DIR + outFile)

        # files = os.listdir(self.INPUT_DIR)
        # for file in files:
        #     if file == '.DS_Store':
        #         continue
        #     dfTmp = self.__comp(file)
        #     df = pd.concat([df,dfTmp],axis=0)
        # # save
        # df.to_csv(self.OUTPUT_DIR + outFile)
        # return dfComp

    def execComp(self, TARGET_FILE=None):
        df = self.__comp(TARGET_FILE)
        # save
        df.to_csv(self.OUTPUT_DIR + "comp_" + TARGET_FILE + "_" + Util.getNow() + ".csv")
        return df

    # 特徴量の比較
    def __comp(self, dfVal, OUTFILE, TARGET_FILE=None):
        dfComp = pd.DataFrame(columns=ImageComp.ARG_INDEX)
        print(dfVal)
        for i, irow in dfVal.iterrows():
            if TARGET_FILE != None and irow['name'] != TARGET_FILE:
                continue
            for j, jrow in dfVal.iterrows():
                if i==j:
                    continue
                psComp = self.__compVal(irow, jrow)
                psComp.to_csv(self.OUTPUT_DIR + OUTFILE, mode='a')
                # dfComp.append(psComp, ignore_index=True)


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
        FEATURE_HIST = self.OUTPUT_DIR + "feature_" + self.NO + "_hist" + ".csv"
        FEATURE_AKAZE = self.OUTPUT_DIR + "feature_" + self.NO + "_AKAZE" + ".csv"
        FEATURE_ORB = self.OUTPUT_DIR + "feature_" + self.NO + "_ORB" + ".csv"
        # Save(hist)
        df_hist = pd.DataFrame(listVal['hist'].T)
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

    # 特徴量の計算
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

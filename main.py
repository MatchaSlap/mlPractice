from modules.utillibs import Util
from modules.imageProc import ImageComp
import os, time
import cv2

# Data
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CUR_DIR + "/data/data-IM/"
INPUT_DIR = BASE_DIR + "input/"
OUTPUT_DIR = BASE_DIR + "output/"

#Time-Start
start = time.time()

# Proc
im = ImageComp(INPUT_DIR,OUTPUT_DIR)
im.execCompAll()

#Time-End
procTime = time.time() - start

# End
print("procTime:",procTime)

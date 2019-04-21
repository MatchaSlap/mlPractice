from modules.utillibs import Util
from modules.imageProc import ImageComp
import time, pathlib
import cv2

# DataPath(pathlib)
CUR_DIR = pathlib.Path(__file__).parent.resolve()
BASE_DIR = CUR_DIR / 'data' / 'data-IM'
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
FEATURE_DIR = BASE_DIR / 'feature'
# DataPath(string)
sINPUT_DIR = str(INPUT_DIR) + "/"
sOUTPUT_DIR = str(OUTPUT_DIR) + "/"
sFEATURE_DIR = str(FEATURE_DIR) + "/"

#Time-Start
start = time.time()

"""
評価手法毎の実装
特徴抽出の記録

"""

# Proc
im = ImageComp(sINPUT_DIR, sOUTPUT_DIR, sFEATURE_DIR)
im.execCompAll()
# im.test()

#Time-End
procTime = time.time() - start

# End
print("procTime:",procTime)

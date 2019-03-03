from datetime import datetime
import json
class Util(object):
    @classmethod
    def getNow(cls):
        return datetime.now().strftime("%Y%m%d%H%M%S")
    @classmethod
    def loadData(self, dataPath):
        with open(dataPath) as f:
            df = json.load(f)
        return df

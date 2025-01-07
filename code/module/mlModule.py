import os
from config import config
from prototype.module import Module
from moduleConfig.mlConfig import MLConfig
from moduleResult.mlResult import MLResult

class MLModule(Module):
    def __init__(self, config:MLConfig):
        super().__init__(config)
        self.mlConfig = config

    def run(self):
        resultFile = f"{config.resultBase}/{self.mlConfig.name}.csv"
        self.resultDict = dict()
        if (os.path.exists(resultFile)):
            with open(resultFile) as fp:
                fp.readline()
                for line in fp:
                    terms = line.strip().split(',')
                    self.resultDict[terms[0]] = terms[1]
    
    def getResult(self, sample):        
        if (sample.query in self.resultDict):
            return MLResult(self.resultDict[sample.query])
        else:
            return None
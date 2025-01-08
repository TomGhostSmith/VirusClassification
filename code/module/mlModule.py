import os
from config import config
from prototype.module import Module
from moduleResult.mlResult import MLResult

class MLModule(Module):
    def __init__(self, shortname):
        self.shortname = shortname
        super().__init__(f'ML-param=contig_most_frequent_{self.shortname}')
        self.baseName = self.moduleName
        
    def run(self):
        resultFile = f"{config.resultBase}/{self.baseName}.csv"
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
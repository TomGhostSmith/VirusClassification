from config import config
from prototype.module import Module
from moduleConfig.mlConfig import mlConfig
from moduleResult.mlResult import MLResult

class MLModule(Module):
    def __init__(self):
        super().__init__()

    def run(self):
        return super().run()
    
    def getResults(self, sampleList):
        resultFile = f"{config.resultBase}/MLResult-{mlConfig.name}.csv"
        resultDict = dict()
        with open(resultFile) as fp:
            fp.readline()
            for line in fp:
                terms = line.strip().split(',')
                resultDict[terms[0]] = terms[1]
        
        for sample in sampleList:
            if (sample.query in resultDict):
                sample.addResult("ML", MLResult(resultDict[sample.query]))


mlModule = MLModule()
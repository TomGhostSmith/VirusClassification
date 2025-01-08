from entity.sample import Sample
from prototype.result import Result

class Module():
    def __init__(self, name):
        self.moduleName = name

    # should check if answer is exists
    # make sure that the answer can be randomly accessed
    def run(self):
        pass

    def getResults(self, sampleList):
        for sample in sampleList:
            if (self.moduleName not in sample.results):
                sample.addResult(self.moduleName, self.getResult(sample))

    # each model should return the result of the sample. If no, return None
    def getResult(self, sample:Sample)->Result:
        pass
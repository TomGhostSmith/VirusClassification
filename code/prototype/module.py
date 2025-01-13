from entity.sample import Sample
from prototype.result import Result

class Module():
    def __init__(self, name):
        self.moduleName = name

    # only run the samples (which is not in the answer)
    # make sure that the answer can be randomly accessed
    def run(self, samples):
        pass

    def getResults(self, sampleList):
        samples = list()
        for sample in sampleList:
            if (self.moduleName not in sample.results):
                samples.append(sample)
        results = self.run(samples)
        for sample, result in zip(samples, results):
            sample.addResult(self.moduleName, result)

    # # each model should return the result of the sample. If no, return None
    # def getResult(self, sample:Sample)->Result:
    #     pass
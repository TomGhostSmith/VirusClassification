from entity.sample import Sample
from prototype.result import Result

class Module():
    def __init__(self, name):
        self.moduleName = name

    def getResults(self, sampleList:list[Sample]):
        samples:list[Sample] = list()
        for sample in sampleList:
            if (self.moduleName not in sample.results):
                samples.append(sample)
        if (len(samples) > 0):
            results = self.run(samples)
            for sample, result in zip(samples, results):
                sample.addResult(self.moduleName, result)

    # only run the samples (which are not in the answer)
    def run(self, samples:list[Sample])->list[Result]:
        pass
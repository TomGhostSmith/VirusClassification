from entity.sample import Sample
from prototype.result import Result
from utils import IOUtils

class Module():
    def __init__(self, name):
        self.moduleName:str = name

    def getResults(self, sampleList:list[Sample], **kwargs):
        samples:list[Sample] = list()
        if kwargs:
            samples = sampleList
        else:
            for sample in sampleList:
                if (self.moduleName not in sample.results):
                    samples.append(sample)
        if (len(samples) > 0):
            results = self.run(samples, **kwargs)
            for sample, result in zip(samples, results):
                sample.addResult(self.moduleName, result)
        pass

    # only run the samples (which are not in the answer)
    def run(self, samples:list[Sample], **kwargs)->list[Result]:
        pass
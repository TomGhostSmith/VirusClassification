from prototype.module import Module
from entity.sample import Sample

class MergeModule(Module):
    def __init__(self, models, name):
        super().__init__(name)
        self.models = models

    def run(self, samples):
        results = list()
        for sample in samples:
            results.append(self.getResult(sample))
        return results
    
    def getResults(self, sampleList):
        for model in self.models:
            model.getResults(sampleList)
        super().getResults(sampleList)

    def getResult(self, sample:Sample):
        availableResults = dict()
        result = None
        source = None
        for model in self.models:
            if (sample.results[model.moduleName] is not None):
                availableResults[model.moduleName] = sample.results[model.moduleName]
        
        if (len(availableResults) == 1):
            source = list(availableResults.keys())[0]
            result = list(availableResults.values())[0]
        elif (len(availableResults) > 1):
            source = self.selectResult(sample, availableResults)
            result = sample.results[source]

        sample.addInfo(self.moduleName, source)
        return result

    # return the model name that result want to use
    def selectResult(self, sample:Sample, availableResults)-> str:
        pass
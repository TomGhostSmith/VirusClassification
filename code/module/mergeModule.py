from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module

class MergeModule(Module):
    def __init__(self, modelList:list[Module], chooseMethod, name:str):
        self.modelList = modelList
        self.modelNames = [model.moduleName for model in modelList]
        self.chooseMethod = chooseMethod
        super().__init__(name)

    def run(self, samples:list[Sample], **kwargs):
        results = dict()
        remainedSamples = samples
        for idx, model in enumerate(self.modelList):
            if (len(remainedSamples) == 0):
                IOUtils.showInfo("All samples predicted. Skip remained models")
                break
            IOUtils.showInfo(f"Merge result: running {model.moduleName} on {len(remainedSamples)} samples")
            model.getResults(remainedSamples)
            unTerminatedSamples = list()
            for sample in remainedSamples:
                res = self.chooseMethod(sample, self.modelNames, idx)
                if res is None:
                    unTerminatedSamples.append(sample)
                else:
                    results[sample.id] = res
            remainedSamples = unTerminatedSamples
            
        resultList = list()
        for sample in samples:
            resultList.append(results.get(sample.id))
        
        return resultList
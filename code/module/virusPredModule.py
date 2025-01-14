from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult
from entity.sample import Sample

class VirusPred(Module):
    def __init__(self, models:list[Module]):
        self.models = models
        names = [model.moduleName for model in models]
        super().__init__('.'.join(names))

    # def run(self):
    #     for model in self.models:
    #         model.run()

    def run(self, samples:list[Sample]):
        unTerminatedSamples = samples
        virus = set()
        for model in self.models:
            model.getResults(unTerminatedSamples)
            s = list()
            for sample in unTerminatedSamples:
                if sample.results[model.moduleName] is not None:
                    virus.add(sample.id)
                else:
                    s.append(sample)
            unTerminatedSamples = s
        
        results = list()
        for sample in samples:
            if sample.id in virus:
                results.append(VirusPredictionResult())
            else:
                results.append(None)

        return results
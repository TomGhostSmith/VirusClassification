from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult

class VirusPred(Module):
    def __init__(self, models:list[Module]):
        self.models = models
        names = [model.moduleName for model in models]
        super().__init__('.'.join(names))

    def run(self, samples:list[Sample], **kwargs):
        unTerminatedSamples = samples
        virus = set()
        for model in self.models:
            IOUtils.showInfo(f'apply {model.moduleName} on {len(unTerminatedSamples)} samples')
            model.getResults(unTerminatedSamples)
            s = list()
            for sample in unTerminatedSamples:
                if sample.results[model.moduleName] is not None and sample.results[model.moduleName][0].score >= 0.5:
                    virus.add(sample.id)
                else:
                    s.append(sample)
            IOUtils.showInfo(f'{len(unTerminatedSamples) - len(s)} samples are confirmed as virus by {model.moduleName}')
            unTerminatedSamples = s
        
        results = list()
        for sample in samples:
            if sample.id in virus:
                results.append([VirusPredictionResult(1, True)])
            else:
                results.append([VirusPredictionResult(0, False)])

        return results
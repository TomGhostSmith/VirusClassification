from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult

class VirusPred(Module):
    def __init__(self, models):
        self.models = models
        names = [model.moduleName for model in models]
        super().__init__('.'.join(names))

    # def run(self):
    #     for model in self.models:
    #         model.run()
    
    def getResults(self, sampleList):
        for model in self.models:
            model.getResults(sampleList)
        super().getResults(sampleList)

    def getResult(self, sample):
        for model in self.models:
            if (sample.results[model.moduleName] is not None):
                return VirusPredictionResult()
        
        return None

    

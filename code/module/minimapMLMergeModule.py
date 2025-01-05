from moduleConfig.minimapMLMergeConfig import MinimapMLMergeConfig
from prototype.mergeModule import MergeModule
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment

class MinimapMLMergeModule(MergeModule):
    def __init__(self, config:MinimapMLMergeConfig):
        super().__init__(config)
        self.minimapMLMergeConfig = config

    def run(self):
        return super().run()
    
    def selectResult(self, sample, availableResults):
        minimapName = self.minimapMLMergeConfig.minimapConfig.name
        mlName = self.minimapMLMergeConfig.mlConfig.name
        resultSource = minimapName
        minimapResult:MinimapResult = availableResults[minimapName]
        minimapAlignment:Alignment = minimapResult.bestAlignment

        if ("positive" in self.minimapMLMergeConfig.factors and minimapAlignment.quality == 0):
            resultSource = mlName
        if ("60" in self.minimapMLMergeConfig.factors and minimapAlignment.quality < 60):
            resultSource = mlName
        if ("completeMatch" in self.minimapMLMergeConfig.factors and minimapAlignment.queryCoverLength < sample.length):
            resultSource = mlName
        if ("singleAlignment" in self.minimapMLMergeConfig.factors and len(minimapResult.alignments) > 1):
            resultSource = mlName
        
        return resultSource
from prototype.mergeModule import MergeModule
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment
from module.minimap import Minimap
from module.minimapThreshRankModule import MinimapThreshRankModule
from module.mlModule import MLModule

class MinimapMLMergeModule(MergeModule):
    def __init__(self, minimap:MinimapThreshRankModule, mlModule:MLModule, factors=['most']):
        self.minimap = minimap
        self.mlModule = mlModule
        self.factors = factors if isinstance(factors, list) else [factors]
        super().__init__([minimap, mlModule], f"{self.minimap.moduleName}.{self.mlModule.moduleName}.minimapML-{"_".join(self.factors)}")

    # def run(self):
    #     return super().run()
    
    def selectResult(self, sample, availableResults):
        minimapName = self.minimap.moduleName
        mlName = self.mlModule.moduleName
        resultSource = minimapName
        minimapResult:MinimapResult = availableResults[minimapName]
        minimapAlignment:Alignment = minimapResult.bestAlignment

        if ("positive" in self.factors and minimapAlignment.quality == 0):
            resultSource = mlName
        if ("60" in self.factors and minimapAlignment.quality < 60):
            resultSource = mlName
        if ("completeMatch" in self.factors and minimapAlignment.queryCoverLength < sample.length):
            resultSource = mlName
        if ("singleAlignment" in self.factors and len(minimapResult.alignments) > 1):
            resultSource = mlName
        
        return resultSource
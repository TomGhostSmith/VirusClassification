from moduleConfig.mergeConfig import mergeConfig
from prototype.module import Module
from module.minimap import minimap
from module.mlModule import mlModule
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment

class MergeModule(Module):
    def __init__(self):
        super().__init__()

    def run(self):
        return super().run()
    
    def getResults(self, sampleList):
        minimap.getResults(sampleList)
        mlModule.getResults(sampleList)

        for sample in sampleList:
            if (len(sample.results) == 1):
                sample.addResult("merge", list(sample.results.values())[0])
                sample.addInfo("mergeSource", list(sample.results.keys())[0])
            elif (len(sample.results) == 2):
                resultSource = 'minimap'
                minimapResult:MinimapResult = sample.results['minimap']
                minimapAlignment:Alignment = minimapResult.bestAlignment
                if ("positive" in mergeConfig.factors and minimapAlignment.quality == 0):
                    resultSource = 'ML'
                if ("60" in mergeConfig.factors and minimapAlignment.quality < 60):
                    resultSource = 'ML'
                if ("completeMatch" in mergeConfig.factors and minimapAlignment.queryCoverLength < sample.length):
                    resultSource = 'ML'
                if ("singleAlignment" in mergeConfig.factors and len(minimapResult.alignments) > 1):
                    resultSource = 'ML'
                
                sample.addResult("merge", sample.results[resultSource])
                sample.addInfo("mergeSource", resultSource)
                    




        
    
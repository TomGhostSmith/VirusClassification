from prototype.mergeModule import MergeModule
from prototype.module import Module
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment
from module.minimap import Minimap
from module.minimapThreshRankModule import MinimapThreshRankModule
from module.mlModule import MLModule
from entity.sample import Sample
from utils import IOUtils

class MinimapMLMergeModule(Module):
    def __init__(self, minimap:MinimapThreshRankModule, mlModule:MLModule, factors=['most']):
        self.minimap = minimap
        self.mlModule = mlModule
        self.factors = factors if isinstance(factors, list) else [factors]
        super().__init__(f"{self.minimap.moduleName}.{self.mlModule.moduleName}.minimapML-{'_'.join(self.factors)}")

    def run(self, samples:list[Sample]):
        resultDict = dict()

        # 1. use minimapThrank with merge condition
        IOUtils.showInfo(f'{len(samples)} samples will be classified')
        unSolvedSamples:list[Sample] = list()
        self.minimap.getResults(samples)
        for sample in samples:
            resultDict[sample.id] = self.getMinimapResult(sample)
            if (resultDict[sample.id] is None):
                unSolvedSamples.append(sample)
        
        # 2. use ml for the remained part
        IOUtils.showInfo(f"{len(samples) - len(unSolvedSamples)} samples are scored by minimap with limit")
        IOUtils.showInfo(f"{len(unSolvedSamples)} samples will be scored by ML model")
        s = list()
        self.mlModule.getResults(unSolvedSamples)
        for sample in unSolvedSamples:
            resultDict[sample.id] = sample.results[self.mlModule.moduleName]
            if (resultDict[sample.id] is None):
                s.append(sample)
        IOUtils.showInfo(f'{len(unSolvedSamples) - len(s)} samples are scored by ML model')
        unSolvedSamples = s

        # 3. if still no result, try to use minimap without merge condition again
        IOUtils.showInfo(f'{len(unSolvedSamples)} samples are waiting for results')
        scoredSamples = 0
        for sample in unSolvedSamples:
            resultDict[sample.id] = sample.results[self.minimap.moduleName]
            if (sample.results[self.minimap.moduleName] is not None):
                scoredSamples += 1
        IOUtils.showInfo(f'{scoredSamples} are scored again by minimap')
        IOUtils.showInfo(f'{len(unSolvedSamples) - scoredSamples} are not scored')
        
        results = list()
        for sample in samples:
            results.append(resultDict[sample.id])

        return results

    def getMinimapResult(self, sample:Sample):    
        minimapName = self.minimap.moduleName
        minimapResult:MinimapResult = sample.results[minimapName]
        if (minimapResult is None):
            return None
        minimapAlignment:Alignment = minimapResult.bestAlignment

        if ("positive" in self.factors and minimapAlignment.quality == 0):
            return None
        if ("60" in self.factors and minimapAlignment.quality < 60):
            return None
        if ("completeMatch" in self.factors and minimapAlignment.queryCoverLength < sample.length):
            return None
        if ("singleAlignment" in self.factors and len(minimapResult.alignments) > 1):
            return None
        
        return minimapResult
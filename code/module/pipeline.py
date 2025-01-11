from prototype.module import Module
from module.minimapMLMergeModule import MinimapMLMergeModule
from module.virusPredModule import VirusPred


class Pipeline(Module):
    def __init__(self, virusPred:VirusPred, minimapMLMerge:MinimapMLMergeModule):
        self.viresPred = virusPred
        self.minimapMLMerge = minimapMLMerge
        super().__init__(f"pipeline_vp={virusPred.moduleName}_merge={minimapMLMerge.moduleName}")

    def run(self):
        # TODO: record what sample are already calculated with confirmed results, so that the remained model does not require to run that
        self.viresPred.run()
        self.minimapMLMerge.run()

    def getResults(self, sampleList):
        # TODO: record what sample are already calculated with confirmed results, so that the remained model does not require to run that
        self.viresPred.getResults(sampleList)
        self.minimapMLMerge.getResults(sampleList)
        super().getResults(sampleList)
    
    def getResult(self, sample):
        if (sample.results[self.viresPred.moduleName] is not None):
            return sample.results[self.minimapMLMerge.moduleName]
        return None
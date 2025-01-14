from prototype.module import Module
from module.minimapMLMergeModule import MinimapMLMergeModule
from module.virusPredModule import VirusPred
from module.minimap import Minimap
from module.minimapThresholdModule import MinimapThresholdModule
from module.minimapThreshRankModule import MinimapThreshRankModule
from module.esm import ESM
from module.mlModule import MLModule
from entity.sample import Sample


class Pipeline(Module):
    def __init__(self, virusPred:VirusPred, virusTaxo:Module):
        self.virusPred = virusPred
        self.virusTaxo = virusTaxo
        super().__init__(f"pipeline_vp={virusPred.moduleName}.merge={virusTaxo.moduleName}")

    def run(self, samples:list[Sample]):
        self.virusPred.getResults(samples)
        virus:list[Sample] = list()
        resultDict = dict()
        for sample in samples:
            if (sample.results[self.virusPred.moduleName] is not None):
                virus.append(sample)
            else:
                resultDict[sample.id] = None

        self.virusTaxo.getResults(virus)
        for sample in virus:
            resultDict[sample.id] = sample.results[self.virusTaxo.moduleName]
        
        results = list()
        for sample in samples:
            results.append(resultDict[sample.id])
        
        return results
    
    def getParams(self):
        param = {
            "pred minimap": "-",
            "ESM": "-",
            "taxo minimap": "-",
            "ML": "-",
            "merge": "-"
        }

        for model in self.virusPred.models:
            if (isinstance(model, MinimapThresholdModule)):
                param["pred minimap"] = [model.reference, model.baseName, model.factors]
            elif (isinstance(model, Minimap)):
                param["pred minimap"] = [model.reference, model.mode, "-"]
            elif (isinstance(model, ESM)):
                param["ESM"] = model.moduleName
        
        if isinstance(self.virusTaxo, MinimapMLMergeModule):
            minimap = self.virusTaxo.minimap
            ml = self.virusTaxo.mlModule
            param['merge'] = self.virusTaxo.factors
        elif isinstance(self.virusTaxo, MinimapThreshRankModule):
            minimap = self.virusTaxo
            ml = None
        elif isinstance(self.virusTaxo, MLModule):
            minimap = None
            ml = self.virusTaxo

        if minimap is not None:
            param['taxo minimap'] = [minimap.reference, minimap.mode, minimap.code]
        if ml is not None:
            param['ML'] = [ml.strategy, str(ml.thresh), ml.gen]

        return param
    
    # 0-2: minimap ref, minimap mode, minimap factor
    # 3: esm
    # 4-6: minimap ref, minimap mode, minimap code
    # 7-9: ML strategy, ML cutoff, ML gen
    # 10: merge
    def getParamList(self):
        param = self.getParams()
        result = list()
        if (isinstance(param['pred minimap'], list)):
            v = param['pred minimap']
            result.append(v[0])
            result.append(v[1])
            if (isinstance(v[2], list)):
                result.append(";".join(v[2]))
            else:
                result.append('-')
        else:
            result += ["-"]*3

        result.append(param["ESM"])

        if isinstance(param['taxo minimap'], list):
            result += param['taxo minimap']
        else:
            result += ['-'] * 3
        
        if (isinstance(param['ML'], list)):
            result += param["ML"]
        else:
            result += ['-']*3
        
        if (isinstance(param['merge'], list)):
            result.append(";".join(param["merge"]))
        else:
            result.append("-")

        return result
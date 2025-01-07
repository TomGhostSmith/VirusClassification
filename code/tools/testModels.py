from moduleConfig.minimapConfig import MinimapConfig
from moduleConfig.minimapMLMergeConfig import MinimapMLMergeConfig
from moduleConfig.mlConfig import MLConfig
from moduleConfig.minimapThreasholdConfig import MinimapThresholdConfig
from module.minimap import Minimap
from module.mlModule import MLModule
from module.minimapMLMergeModule import MinimapMLMergeModule
from module.minimapThreasholdModule import MinimapThreasholdModule
from module.virusPredModule import VirusPred
from module.esm import ESM

from entity.dataset import Dataset
from entity.evaluator import Evaluator
from utils import IterUtils
from config import config

def setMinimap(param):
    minimapConfig = MinimapConfig()
    ref, mode = param
    minimapConfig.mode=mode
    minimapConfig.reference = ref
    return Minimap(minimapConfig)

def setML(param):
    mlConfig = MLConfig()
    mlConfig.shortname = param
    # mlConfig.shortname = "t33_512"
    # mlConfig.shortname = "family_finetune_t33_256"
    return MLModule(mlConfig)

def setMerge(param):
    minimap, mlModule, factors = param
    minimapMLConfig = MinimapMLMergeConfig(minimap, mlModule)
    minimapMLConfig.factors = factors
    return MinimapMLMergeModule(minimapMLConfig)


def testModels():
    models = list()

    minimapParams = [
        ('VMRv4', 'ont')
        ]

    mlParamls = [
        't33_512',
        # 'family_finetune_t33_256'
        ]
    
    mergeFactors = [
        ['most'],
        ['positive'],
        ['60'],
        ['completeMatch'],
        ['positive', 'completeMatch'],
        ['60', 'completeMatch'],
        ['singleAlignment'],
        ['positive', 'singleAlignment'],
        ['60', 'singleAlignment'],
        ['completeMatch', 'singleAlignment'],
        ['positive', 'completeMatch', 'singleAlignment'],
        ['60', 'completeMatch', 'singleAlignment'],
    ]

    # for minimapP in minimapParams:
    #     minimap = setMinimap(minimapP)
    #     for mlP in mlParamls:
    #         ML = setML(mlP)
    #         for factor in mergeFactors:
    #             mergeModel = setMerge((minimap, ML, factor))
    #             models.append(mergeModel)

    esmModel = ESM()

    for minimapP in minimapParams:
        for mergeFactor in mergeFactors:
            ref, mode = minimapP
            c = MinimapThresholdConfig()
            c.factors = mergeFactor
            c.mode = mode
            c.reference = ref
            minimapTh = MinimapThreasholdModule(c)
            # models.append(minimapTh)
            models.append(VirusPred([minimapTh, esmModel]))

    evaluator = Evaluator(models)
    # evaluator.evaluate('all')
    # evaluator.evaluateAllMinorSets(Dataset("refseq_2024_test", config.minorDatasetRanks))
    # evaluator.checkVirusFilter(Dataset("refseq_2024_test", config.minorDatasetRanks))
    evaluator.checkVirusFilter(Dataset("genbank_2024_test", config.minorDatasetRanks))


def getBasicResults():
    models = list()

    minimapParams = [
        ('VMRv4', 'ont')
        ]

    mlParamls = [
        't33_512',
        # 'family_finetune_t33_256'
        ]

    for minimapP in minimapParams:
        minimap = setMinimap(minimapP)
        models.append(minimap)

    # for mlP in mlParamls:
    #     ML = setML(mlP)
    #     models.append(ML)



    evaluator = Evaluator(models)
    # evaluator.evaluate('all')
    # evaluator.evaluate('withResult')
    # evaluator.compare('intersection')
    # evaluator.evaluateAllMinorSets(Dataset("refseq_2024_test", config.minorDatasetRanks[1:]))
    evaluator.evaluateAllMinorSets(Dataset("genbank_2024_test", config.minorDatasetRanks[1:]))

def main():
    # dataset = Dataset("Challenge")
    # dataset = Dataset("gen2")
    # dataset = Dataset("gen2_fold2020")
    # dataset = Dataset("gen2_fold2024")
    # dataset = Dataset("refseq_2024_test", config.minorDatasetRanks)
    # dataset = Dataset("refseq_2024_test", "genus")
    # dataset = Dataset("genbank_2024_test", config.minorDatasetRanks)
    dataset = Dataset("genbank_2024_test", "genus")
    IterUtils.iterDatasets(dataset, testModels)
    # IterUtils.iterDatasets(dataset, getBasicResults)



if (__name__ == '__main__'):
    main()
from module.minimap import Minimap
from module.mlModule import MLModule
from module.minimapMLMergeModule import MinimapMLMergeModule
from module.minimapThresholdModule import MinimapThresholdModule
from module.virusPredModule import VirusPred
from module.esm import ESM
from module.minimapThreshRankModule import MinimapThreshRankModule
from module.pipeline import Pipeline

from entity.dataset import Dataset
from entity.evaluator import Evaluator
from utils import IterUtils
from config import config


def testModels(dataset:Dataset):
    models = list()

    minimapParams = [
        ('VMRv4', 'ont')
        ]

    mlParamls = [
        't33_512',
        'family_finetune_t33_256'
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

    esmConfigs = [
        "150M_256",
        "150M_512",
        "650M_256",
        "650M_256_merge"
    ]

    # for minimapP in minimapParams:
    #     for mlP in mlParamls:
    #         for factor in mergeFactors:
    #             models.append(MinimapMLMergeModule(Minimap(*minimapP), MLModule(mlP), factors=factor))

    # for minimapP in minimapParams:
    #     for mergeFactor in mergeFactors:
    #         models.append(MinimapThresholdModule(*minimapP, factors=mergeFactor))
    # for esmP in esmConfigs:
    #     models.append(ESM(esmP))
    # for esmP in esmConfigs:
    #     for minimapP in minimapParams:
    #         for mergeFactor in mergeFactors:
    #             models.append(VirusPred([MinimapThresholdModule(*minimapP, factors=mergeFactor), ESM(esmP)]))

    evaluator = Evaluator(models)
    evaluator.evaluate('all', dataset)
    # evaluator.checkVirusFilter(dataset)


def getBasicResults(dataset:Dataset):
    models = list()

    minimapParams = [
        ('VMRv4', 'ont')
        ]

    mlParamls = [
        # 't33_512',
        # 'family_finetune_t33_256'
        ("topdown", 0.45)
        ]
    
    thRank = {
        "": "f",
        "pos": "g",
        "60": "g",
        "cm": "g",
        "sa": "g",
        "sa_cm": "g",
        "pos_sa": "g",
        "60_sa": "g",
        # "pos_cm": "g",
        # "pos_cm_sa": "g"
    }

    # for minimapP in minimapParams:
    #     models.append(Minimap(*minimapP))
    
    # for minimapP in minimapParams:
    #     models.append(MinimapThreshRankModule(*minimapP, limitOutputDict=thRank))

    for mlP in mlParamls:
        models.append(MLModule(*mlP))



    evaluator = Evaluator(models)
    evaluator.evaluate('all', dataset)
    # evaluator.evaluate('withResult', dataset)
    # evaluator.compare('intersection', dataset)


def testPipeline(dataset:Dataset):

    thRank = {
        "": "f",
        "pos": "g",
        "60": "g",
        "cm": "g",
        "sa": "g",
        "sa_cm": "g",
        "pos_sa": "g",
        "60_sa": "g",
        "pos_cm": "g",
        "pos_cm_sa": "g"
    }

    strategies = ['topdown', 'topdownExt', 'topdownPreserve', 'bottomup','highest']
    cutoffs = [0.4, 0.45, 0.5]
    mlgens = ['gen1', 'gen2']

    mlParams = list()
    for strategy in strategies:
        for cutoff in cutoffs:
            for gen in mlgens:
                mlParams.append((strategy, cutoff, gen))

    esmList = [
        "650M_256_merge",
        "150M_512"
    ]

    mergefactors = [
        ['most'],
        # ['positive'],
        ['60'],
        # ['completeMatch'],
        # ['positive', 'completeMatch'],
        ['60', 'completeMatch'],
        # ['singleAlignment'],
        # ['positive', 'singleAlignment'],
        # ['60', 'singleAlignment'],
        # ['completeMatch', 'singleAlignment'],
        # ['positive', 'completeMatch', 'singleAlignment'],
        ['60', 'completeMatch', 'singleAlignment'],
    ]

    minimapPredFactor = [
        # ['most'],
        # ['positive'],
        # ['60'],
        # ['completeMatch'],
        # ['positive', 'completeMatch'],
        ['60', 'completeMatch'],
        # ['singleAlignment'],
        # ['positive', 'singleAlignment'],
        # ['60', 'singleAlignment'],
        # ['completeMatch', 'singleAlignment'],
        # ['positive', 'completeMatch', 'singleAlignment'],
        # ['60', 'completeMatch', 'singleAlignment'],
    ]

    models = list()

    for predFactor in minimapPredFactor:
        minimapThrank = MinimapThreshRankModule("VMRv4", limitOutputDict=thRank)
        minimapThreshold = MinimapThresholdModule("VMRv4", factors=predFactor)
        for esm in esmList:
            virusPred = VirusPred([minimapThreshold, ESM(esm)])
            for s in mlParams:
                ml = MLModule(*s)
                for fac in mergefactors:
                    models.append(Pipeline(virusPred, MinimapMLMergeModule(minimapThrank, ml, factors=fac)))

        for esm in esmList:
            virusPred = VirusPred([minimapThreshold, ESM(esm)])
            models.append(Pipeline(virusPred, minimapThrank))

        for esm in esmList:
            virusPred = VirusPred([minimapThreshold, ESM(esm)])
            for s in mlParams:
                ml = MLModule(*s)
                models.append(Pipeline(virusPred, ml))
    
    evaluator = Evaluator(models)
    evaluator.evaluate('all', dataset)





def main():
    # dataset = Dataset("Challenge")
    # dataset = Dataset("gen2")
    # dataset = Dataset("gen2_fold2020")
    # dataset = Dataset("gen2_fold2024")
    dataset = Dataset("refseq_2024_test", config.minorDatasetRanks[0:])
    # dataset = Dataset("refseq_2024_test", "genus")
    # dataset = Dataset("genbank_2024_test", config.minorDatasetRanks[1:])
    # dataset = Dataset("genbank_2024_test", "genus")
    # IterUtils.iterDatasets(dataset, testModels, splitReports=True)
    # IterUtils.iterDatasets(dataset, testModels, splitReports=False)
    # IterUtils.iterDatasets(dataset, getBasicResults, splitReports=True)
    # IterUtils.iterDatasets(dataset, getBasicResults, splitReports=False)
    IterUtils.iterDatasets(dataset, testPipeline, splitReports=False)



if (__name__ == '__main__'):
    main()
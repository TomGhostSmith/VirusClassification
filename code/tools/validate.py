import json
from config import config

def processValidation():
    from entity.taxoTree import taxoTree
    resDict = dict()
    # refID2Node = dict()
    # with open('/Data/ICTVData/reference/VMRv4.tar.gz') as fp:
    #     refID2Taxo = json.load(fp)
    with open('./working/dataset_challenge_all.fasta.matched_sequences_ID.csv') as fp:
        fp.readline()
        for line in fp:
            refID, dataID = line.strip().split(',')
            resDict[dataID] = taxoTree.getTaxoNodeFromICTV(ICTVID=refID).ICTVNode.name
    
    with open('./working/answer.json', 'wt') as fp:
        json.dump(resDict, fp, indent=2)
    
    with open('./working/test', 'wt') as fp:
        fp.writelines([k + '\n' for k in resDict.keys()])


from config import config

import argparse
import os
config.resultRoot = "/Data/ICTVPublish2/results"
config.modelRoot = "/Data/ICTVPublish2/model"
config.dataRoot = "/Data/ICTVPublish2/"
config.updatePath()

def main():
    from module.pipeline import Pipeline
    from module.virusPredModule import VirusPred
    from module.minimapMLMergeModule import MinimapMLMergeModule
    from module.minimapThreshRankModule import MinimapThreshRankModule
    from module.minimapThresholdModule import MinimapThresholdModule
    from module.mlModule import MLModule
    from module.esm import ESM
    from entity.sample import Sample
    from entity.modelRunnder import ModelRunnder
    from entity.dataset import Dataset

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
        "pos_cm_sa": "g",
        "60_cm": 'g',
        "60_cm_sa": 'g'
    }
    pipeline = Pipeline(
        VirusPred([
            MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
            ESM()]),
            # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
            # MLModule('bottomup', 0.45, '1011000')
        MinimapMLMergeModule(
            MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
            # MLModule('topdown', 0.45, '1111000')
            MLModule('highest', 0.45, '1011000')
            # MLModule('bottomup', 0.45, '1011000')
            )
        )
    
    # evaluator = ModelRunnder([pipeline], Dataset("refseq_2024_test"))
    # evaluator = ModelRunnder([pipeline], Dataset("genbank_2024_test"))
    evaluator = ModelRunnder([pipeline], Dataset("Challenge"))
    # evaluator = ModelRunnder([pipeline], fastaPath)

    config.datasetBase = "/Data/ICTVPublish2/dataset/Challenge"
    evaluator.run()


# config.updatePath()
# processValidation()

main()
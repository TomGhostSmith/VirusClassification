from config import config

import argparse
import os

def main(MLstrategy, lowestRank, fastaPath):
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
        "60_cm": lowestRank[0],
        "60_cm_sa": lowestRank[0]
    }
    pipeline = Pipeline(
        VirusPred([
            MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
            ESM()]),
            # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        MinimapMLMergeModule(
            MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
            MLModule('topdown', 0.45, '1111000')
            # MLModule('highest', 0.45, '1011000')
            )
        )
    
    # evaluator = ModelRunnder([pipeline], Dataset("refseq_2024_test"))
    # evaluator = ModelRunnder([pipeline], Dataset("genbank_2024_test"))
    # evaluator = ModelRunnder([pipeline], Dataset("Challenge"))
    evaluator = ModelRunnder([pipeline], fastaPath)
    evaluator.run()
    
            


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description="VirTaxonomer: A deep learning based model to taxomize virus")
    parser.add_argument('--input', help="the fasta file to process", required=True)
    parser.add_argument('--data', help='the output folder', required=True)
    # parser.add_argument('--model', help='the model folder')
    parser.add_argument('--ML', help="machine learning model strategy", default='highest')
    parser.add_argument('--restrict', help="restrict lowest prediction rank", default='genus')
    args = parser.parse_args()
    input = args.input
    data = args.data
    ML = args.ML
    restrict = args.restrict
    
    if (not (input.endswith('fasta') or input.endswith('fa'))):
        raise ValueError('The file you input seems not a fasta file')
    
    if (not os.path.isdir(data)):
        raise ValueError('The model path should be a folder')
    
    if (ML not in ['highest', 'bottomup']):
        raise ValueError('The ML strategy should be either "highest" or "bottomup"')
    
    if (restrict not in ['genus', 'species']):
        raise ValueError('The restrict rank should be either "genus" or "species"')

    fileName = os.path.splitext(os.path.basename(input))[0]
    config.outputName = fileName + '.csv'
    config.dataRoot = data
    config.updatePath()

    main(ML, restrict, input)
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

def main():
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
    pipeline = Pipeline(
        VirusPred([
            MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
            ESM()]),
            # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        MinimapMLMergeModule(
            MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
            # MLModule('topdown', 0.45, '1111000')
            MLModule('highest', 0.45, '1011000')
            )
        )
    
    # evaluator = ModelRunnder([pipeline], Dataset("refseq_2024_test"))
    # evaluator = ModelRunnder([pipeline], Dataset("genbank_2024_test"))
    evaluator = ModelRunnder([pipeline], Dataset("Challenge"))
    evaluator.run()
    
            


if (__name__ == '__main__'):
    main()
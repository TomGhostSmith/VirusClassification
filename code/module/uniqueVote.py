# reconstructed
import os
import sys
import json
import time
import shutil
import subprocess

from prototype.module import Module
from entity.taxoTree import taxoTree
from config import config
from utils import IOUtils
from entity.sample import Sample
from moduleResult.plainResult import PlainResult
from module.marker import Marker
from module.mlModule import MLModule
from module.esmTaxo import ESMTaxo
from moduleResult.diamondAlignment import DiamondAlignment


class UniqueVote(Module):
    def __init__(self, trainset, strategy, mlModel):
        super().__init__(f"UniqueVote-{trainset}-{strategy}-{mlModel}")
        if (trainset not in ["VMRv4", "VMRv4_ML_train"]):
            raise ValueError("Unsupported training set")
        self.trainset = trainset
        if (strategy not in ["both", "ml", "marker"]):   # full_ml: voting with marker + ml. ml: use ml only when marker is not available
            raise ValueError("Unsupported strategy")
        self.strategy = strategy

        modelParams = {
            "realm_esm2_t33_256": (256, f"{config.modelRoot}/realm/esm2_t33_256", "facebook/esm2_t33_650M_UR50D", 29),
            "realm_esm2_t33_512": (512, f"{config.modelRoot}/realm/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 29),
            "kingdom_esm2_t33_256": (256, f"{config.modelRoot}/kingdom/esm2_t33_256", "facebook/esm2_t33_650M_UR50D", 40),
            "kingdom_esm2_t33_512": (512, f"{config.modelRoot}/kingdom/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 40),
            "phylum_esm2_t33_256": (256, f"{config.modelRoot}/phylum/esm2_t33_256", "facebook/esm2_t33_650M_UR50D", 51),
            "phylum_esm2_t33_512": (512, f"{config.modelRoot}/phylum/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 51),
            "class_esm2_t33_256": (256, f"{config.modelRoot}/class/esm2_t33_256", "facebook/esm2_t33_650M_UR50D", 76),
            "class_esm2_t33_512": (512, f"{config.modelRoot}/class/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 76),
            "order_esm2_t33_512": (512, f"{config.modelRoot}/order/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 981),
            "family_esm2_t33_512": (512, f"{config.modelRoot}/family/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 1129),
            "family_esm2_t33_512_enlarge": (512, f"{config.modelRoot}/family/esm2_t33_512_enlarge", "facebook/esm2_t33_650M_UR50D", 1129),
            "genus_esm2_t33_256": (256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune", "facebook/esm2_t33_650M_UR50D", 3523),
            "genus_esm2_t33_256_enlarge": (256, f"{config.modelRoot}/genus/esm2_t33_256_enlarge_genus", "facebook/esm2_t33_650M_UR50D", 3523),
        }
        if (mlModel not in modelParams):
            raise ValueError("Unsupported mlModel")
        self.params = [mlModel, *modelParams[mlModel]]
        self.mlName = mlModel
        self.rank = mlModel[:mlModel.find("_")]

    def run(self, samples):
        markerModule = Marker(self.trainset, "vote", coverage=80, identity=50)
        markerModule.getResults(samples)

        esmTaxo = ESMTaxo(*self.params[:-1], pooling="sum", rank=self.rank)
        esmTaxo.run(samples, keepProb=True)  # should not use getResults because we want to get prob

        taxoLabels = esmTaxo.class_names
        targetRankLevel = config.rankLevels[self.rank]

        results = []

        thresh = 3

        markerLCAs = markerModule.getLCAs()
        key = f"{self.mlName}_prob"

        for sample in samples:
            votes = {taxo: 0 for taxo in taxoLabels}
            for protein in sample.proteins:
                markerPred:list[DiamondAlignment] = protein.results[markerModule.baseName]
                aligned = False
                for alignment in markerPred[:thresh]:
                    taxo = markerLCAs[alignment.ref]
                    node = taxoTree.ICTVTree.nodes[taxo]
                    thisRankLevel = config.rankLevels[node.rank]
                    if thisRankLevel > targetRankLevel:  # species
                        for n in node.path:
                            if config.rankLevels[n.rank] == targetRankLevel:
                                pred = n.name
                        aligned = True
                        break
                    elif thisRankLevel == targetRankLevel:   # genus
                        pred = taxo
                        aligned = True
                        break
                    elif (self.strategy != "ml"):  # family or above
                        pred = taxo
                        if (pred in votes):
                            votes[pred] += alignment.similarity/100
                        else:
                            votes[pred] = alignment.similarity/100
                    

                if (aligned == True and self.strategy != "marker"):  # add ml voting
                    scores = protein.info[f"{self.mlName}_prob"].tolist()
                    rawScores = {taxo: score for taxo, score in zip(taxoLabels, scores)}
                    tops = sorted(list(rawScores.items()), key=lambda x:x[1], reverse=True)
                    for taxo, score in tops[:thresh]:
                        if (taxo in votes):
                            votes[taxo] += score
                        else:
                            votes[taxo] = score

                protein.info.pop(key)

            # print(votes)
            # with open("working/dump.json", "wt") as fp:
            #     json.dump(votes, fp, indent=2)
            totalVotes = sum(votes.values())
            if (len(votes) > 0 and totalVotes > 0):
                winner, maxVotes = max(votes.items(), key=lambda x:x[1])
                results.append(PlainResult(winner, maxVotes/totalVotes))
            else:
                results.append(None)

        return results
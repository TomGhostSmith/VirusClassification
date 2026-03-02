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
from module.diamond import Diamond
from moduleResult.diamondResult import DiamondResult
from moduleResult.diamondAlignment import DiamondAlignment


class UniqueVote(Module):
    def __init__(self, trainset:str, strategy:str, mlModel:str, alignmentMethod:str, pooling:str):
        super().__init__(f"UniqueVote-{trainset}-{strategy}-{mlModel}-{alignmentMethod}-{pooling}")
        if (trainset not in ["VMRv4", "VMRv4_ML_train"]):
            raise ValueError("Unsupported training set")
        self.trainset = trainset

        if (alignmentMethod not in ["marker", "diamond"]):
            raise ValueError("Unsupported alignment method")
        self.alignmentMethod = alignmentMethod

        if (pooling not in ["vote", "sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        self.pooling = pooling

        # strategy: add which to votes when the protein is aligned in this rank
        # 0: do not add any votes
        # 1: add ml to votes
        # 2: add alignment to votes
        # 3: add ml and alignment to votes
        if len(strategy) != 9:
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
            "genus_esm2_t33_1022": (1022, f"{config.modelRoot}/genus/esm2_t33_650M_UR50D_MAXLEN_1022_bs2x4_accum2_lr3e-5_ep10", "facebook/esm2_t33_650M_UR50D", 3523),
            "genus_esm2_t33_1022_2": (1022, f"{config.modelRoot}/genus/esm2_t33_650M_UR50D_MAXLEN_1022_bs2x4_accum2_lr3e-5_ep10", "facebook/esm2_t33_650M_UR50D", 3523),
            "genus_esm2_t33_1022_2": (1022, f"{config.modelRoot}/genus/esm2_t33_650M_UR50D_MAX_LENGTH_1022_per_device_batch_size_2_num_train_epochs_10_save_steps_10000_lr_3e-5_save_steps_10000", "facebook/esm2_t33_650M_UR50D", 3523),
            "genus_esm2_t33_1022_enlarge": (1022, f"{config.modelRoot}/genus/esm2_t33_650M_UR50D_MAX_LENGTH_1022_per_device_batch_size_2_num_train_epochs_10_save_steps_10000_lr_3e-5_save_steps_10000", "facebook/esm2_t33_650M_UR50D", 3523),
        }
        if (mlModel not in modelParams):
            raise ValueError("Unsupported mlModel")
        self.params = [mlModel, *modelParams[mlModel]]
        self.mlName = mlModel
        self.rank = mlModel[:mlModel.find("_")]


    def run(self, samples, **kwargs):
        if (self.alignmentMethod == "diamond"):
            alignmentModel = Diamond(self.trainset, "vote")
        elif (self.alignmentMethod == "marker"):
            alignmentModel = Marker(self.trainset, "vote", coverage=80, identity=50)
            self.markerLCAs = alignmentModel.getLCAs()
        alignmentModel.run(samples, keepProteinRes=True)
        self.alignmentModelName = alignmentModel.moduleName

        esmTaxo = ESMTaxo(*self.params[:-1], pooling="sum", rank=self.rank)
        esmTaxo.run(samples, keepProb=True)  # should not use getResults because we want to get prob
        self.esmTaxoName = esmTaxo.moduleName
        self.taxoLabels = taxoTree.taxaNames[self.rank]

        results = [self.getResult(sample) for sample in samples]
        return results
    
    def getResult(self, sample:Sample):
        ranks = ["realm", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
        rankLevels = [0] + [config.rankLevels[r] for r in ranks]

        votes = {taxo: 0 for taxo in self.taxoLabels}
        for protein in sample.proteins:
            res:list[DiamondResult] = protein.results[self.alignmentModelName]

            # get best alignment rank
            maxRank = 0
            if res:
                for r in res:
                    node = r.node
                    thisRankLevel = config.rankLevels[node.ICTVNode.rank]
                    if thisRankLevel > maxRank:
                        maxRank = thisRankLevel
            
            # get mode
            mode = 0
            for idx, rankLevel in reversed(list(enumerate(rankLevels))):
                if maxRank >= rankLevel:
                    mode = int(self.strategy[idx])
                    break
            
            useAlignment = mode & 0b10
            useML = mode & 0b01

            # get alignment candidates
            if (useAlignment):
                alignmentCandidates = [(r.node.ICTVName, r.alignment.similarity) for r in res]
                alignmentCandidates = sorted(list(alignmentCandidates), key=lambda x:x[1], reverse=True)
                self.updateVotes(votes, alignmentCandidates)
            
            if (useML):
                mlCandidates = zip(self.taxoLabels, protein.info[f"{self.esmTaxoName}_prob"].tolist())
                mlCandidates = sorted(list(mlCandidates), key=lambda x:x[1], reverse=True)
                self.updateVotes(votes, mlCandidates)

            protein.info.pop(f"{self.esmTaxoName}_prob")

        totalVotes = sum(votes.values())
        if (len(votes) > 0 and totalVotes > 0):
            votes = sorted(votes.items(), key=lambda x:x[1], reverse=True)
            return [PlainResult(w, v/totalVotes) for w, v in votes]
        else:
            return None

    def updateVotes(self, votes, candidates):
        if (self.pooling == "vote"):
            if (len(candidates) > 0):
                bestName, _ = candidates[0]
                if bestName in votes:
                    votes[bestName] += 1
                else:
                    votes[bestName] = 1
        else:
            if (self.pooling.startswith("top")):
                thresh = int(self.pooling[3:])
                candidates = candidates[:thresh]
            # else: pooling == sum, then use all of them

            for name, score in candidates:
                if name in votes:
                    votes[name] += score
                else:
                    votes[name] = score
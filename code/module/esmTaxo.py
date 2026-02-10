# reconstructed
import os
import json
import pandas
from concurrent.futures import ProcessPoolExecutor

from config import config
from prototype.module import Module
from moduleResult.plainResult import PlainResult
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from module.esmRunner import ESMRunner
from tqdm import tqdm

from entity.taxoTree import taxoTree
from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class ESMTaxo(Module):
    def __init__(self, modelName, maxLen, modelFolder, baseModelFolder, rank, pooling, batchSize=None):
        super().__init__(f"ESM_taxo_{modelName}_{pooling}")
        self.name = modelName
        if (pooling not in ["vote", "sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        self.pooling = pooling
        self.maxLen = maxLen
        self.modelFolder = modelFolder
        self.baseModelFolder = baseModelFolder
        
        self.batchSize = batchSize
        self.rank = rank

        self.class_names = taxoTree.taxaNames[self.rank]

    def run(self, samples:list[Sample], keepProb=False, keepVotes=False, keepProteinRes=False, **kwargs)->list[PlainResult]:
        NucleotideUtils.extractProtein(samples)
        proteinsToRun:list[ProteinSample] = list()
        for sample in samples:
            for protein in sample.proteins:
                proteinsToRun.append(protein)

        model = ESMRunner(self.name, self.maxLen, self.modelFolder, self.baseModelFolder, len(self.class_names), self.batchSize)
        model.run(proteinsToRun, getProb=True)
        key = f"{self.name}_prob"

        results = [self.getResult(sample, keepVotes, keepProteinRes) for sample in samples]

        if (keepProb):
            for p in proteinsToRun:
                p.info[f"{self.moduleName}_prob"] = p.info.pop(key, None)
        else:
            for p in proteinsToRun:
                p.info.pop(key, None)
        return results
    
    def getResult(self, sample:Sample, keepVotes, keepProteinRes):
        if (self.pooling == "vote"):
            votes = {n: 0 for n in self.class_names if "Unknown" not in n}
            for protein in sample.proteins:
                scores = protein.info[f"{self.name}_prob"]
                rawScores = [(taxo, score) for taxo, score in zip(self.class_names, scores)]
                tops = sorted(rawScores, key=lambda x:x[1], reverse=True)
                if (keepProteinRes):
                    protein.addResult(self.moduleName, [PlainResult(p, s) for p, s in rawScores if "Unknown" not in p])
                bestTaxo = tops[0][0]
                if ("Unknown" not in bestTaxo):
                    votes[bestTaxo] += 1
        elif (self.pooling == "sum" or self.pooling.startswith("top")):
            if self.pooling.startswith("top"):
                thresh = int(self.pooling[3:])
            else:
                thresh = None
            votes = {n: 0 for n in self.class_names if "Unknown" not in n}
            for protein in sample.proteins:
                scores = protein.info[f"{self.name}_prob"].tolist()
                rawScores = [(taxo, score) for taxo, score in zip(self.class_names, scores)]
                tops = sorted(rawScores, key=lambda x:x[1], reverse=True)
                if (keepProteinRes):
                    protein.addResult(self.moduleName, [PlainResult(p, s) for p, s in rawScores if "Unknown" not in p])
                for taxo, score in tops[:thresh]:
                    if ("Unknown" not in taxo):
                        votes[taxo] += score

        totalVotes = sum(votes.values())    # If pooling method == "sum", the totalVotes will be 1 * len(proteins) (not considering "Unknown" labels)
        if (totalVotes > 0):
            if (keepVotes):
                sample.info[f"{self.moduleName}_votes"] = {k: v/totalVotes for k, v in votes.items()}
            vs = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            results = [PlainResult(n, v/totalVotes) for n, v in vs]
            return results
        else:
            return None
        
def getProbs(name, maxLen, modelFolder, baseModelFolder, n_class, batchSize, proteinsToRun):
    model = ESMRunner(name, maxLen, modelFolder, baseModelFolder, n_class, batchSize)
    model.run(proteinsToRun, getProb=True)
    key = f"{name}_prob"
    results = [p.info[key] for p in proteinsToRun]
    return results
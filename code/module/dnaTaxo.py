import os
import json
import pandas

from prototype.module import Module
from entity.sample import Sample
from moduleResult.plainResult import PlainResult
from config import config
from utils import IOUtils
from entity.taxoTree import taxoTree

class DNATaxo(Module):
    def __init__(self, modelName, rank):
        super().__init__(f"DNA_taxo_{modelName}")
        self.name = modelName
        self.rank = rank

        self.class_names = taxoTree.taxaNames[rank]

        self.cacheProbFile = f"{config.cacheResultFolder}/DNA_taxo_{modelName}_prob.tmp"
        self.cacheProbIndex = f"{config.cacheResultFolder}/DNA_taxo_{modelName}_prob.json"

        self.cachedSamples_prob = {"nextOffset": 0}
        self.nextOffset_prob = 0

    def run(self, samples:list[Sample], keepVotes=False, **kwargs):
        if (os.path.exists(self.cacheProbIndex)):
            with open(self.cacheProbIndex) as fp:
                self.cachedSamples_prob = json.load(fp)
                self.nextOffset_prob = self.cachedSamples_prob["nextOffset"]

        fp = open(self.cacheProbFile)
        results = [self.getResult(sample, fp, keepVotes) for sample in samples]
        fp.close()

        return results

    def getResult(self, sample:Sample, fp, keepVotes):
        offset = self.cachedSamples_prob[sample.id]
        fp.seek(offset)
        line = fp.readline().strip('\n')
        probText = line[line.find('\t')+1:]
        probs = IOUtils.decodeBase64(probText)
        predictions = [PlainResult(pred, prob) for pred, prob in sorted(zip(self.class_names, probs), key=lambda x:x[1], reverse=True) if prob > 0 and "Unknown" not in pred]
        if (keepVotes):
            votes = {pred: prob for pred, prob in zip(self.class_names, probs)}
            sample.info[f"{self.moduleName}_votes"] = votes
        return predictions

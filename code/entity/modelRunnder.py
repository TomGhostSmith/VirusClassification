import os
import json
from Bio import SeqIO

from config import config
from entity.sample import Sample
from module.pipeline import Pipeline
from utils import IOUtils
from prototype.module import Module

class ModelRunnder():
    def __init__(self, model:Module, ):
        self.model = model

    def run(self, inputPath, outputPath):
        samples:list[Sample] = IOUtils.loadSamples(inputPath)
        self.model.getResults(samples)

        resultDict = dict()
        resultList = list()
        for sample in samples:
            res = sample.results[self.model.moduleName]
            if (res is not None):
                resultDict[sample.id] = res.node.ICTVNode.name
                resDict = dict()
                lastScore = 0
                for n in reversed(res.node.ICTVNode.path[1:]):  # skip superkingdom
                    if n.rank in res.scores:
                        lastScore = res.scores[n.rank]
                        resDict[n.rank] = (n.name, lastScore)
                    else:
                        resDict[n.rank] = (n.name, lastScore)
                resultList.append((sample.id, resDict))
            else:
                resultDict[sample.id] = 'no result'
                resultList.append((sample.id, dict()))

        # generate submit-used
        lines = list()
        
        lines.append("\t".join(config.resultCSVRanks) + '\n')
        resultList = sorted(resultList, key=lambda t:t[0])
        for id, resDict in resultList:
            text = [id]
            for rank in config.resultRanks:
                if rank in resDict:
                    name, score = resDict[rank]
                else:
                    name, score = 'N/A', 'N/A'
                text.append(name)
                text.append(str(score))
            lines.append('\t'.join(text) + '\n')
        
        with open(outputPath, 'wt') as fp:
            fp.writelines(lines)
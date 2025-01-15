import os
import json
from Bio import SeqIO

from config import config
from entity.dataset import Dataset
from entity.sample import Sample
from module.pipeline import Pipeline

class ModelRunnder():
    def __init__(self, models, dataPath):
        def load(ids=None):
            queryFile = f"{config.datasetBase}/{config.datasetName}.fasta"
            sampleList = list()
            for record in SeqIO.parse(queryFile, "fasta"):
                if ids is None or record.id in ids:
                    sampleList.append(Sample(seq=record))
            return sampleList
        
        self.models = models
        self.samples:list[Sample] = list()

        if (isinstance(dataPath, Dataset)):
            config.majorDataset = dataPath.majorDataset
            config.updatePath()
            ids = list()
            for minor in dataPath.minorDatasets:
                with open(f"{config.datasetBase}/minorDatasets/{minor}") as fp:
                    for line in fp:
                        ids.append(line.strip())
            ids = set(ids) if len(ids) > 0 else None
            self.samples += load(ids)
        else:
            for record in SeqIO.parse(dataPath, "fasta"):
                self.samples.append(Sample(seq=record))

    def run(self):
        for model in self.models:
            model.getResults(self.samples)

        if (config.outputName is None):
            if (not os.path.exists(f"{config.tempFolder}/resCount")):
                pipelineIndex = 0
            else:
                with open(f"{config.tempFolder}/resCount") as fp:
                    pipelineIndex = int(fp.readline().strip())
            
            with open(f"{config.tempFolder}/resCount", 'wt') as fp:
                fp.write(str(pipelineIndex + len(self.models)))

            pipelineResTitle = ["pred minimap ref", "pred minimap mode", "pred minimap factor",
                                "esm",
                                "taxo minimap ref", "taxo minimap mode", "taxo minimap code",
                                "ML strategy", "ML cutoff", "ML gen",
                                "merge",
                                "pipelineIndex"]

            requireTitle = not os.path.exists(f"{config.resultBase}/pipelines.csv")
            pipelineNameFP = open(f"{config.resultBase}/pipelines.csv", 'at')

            if (requireTitle):
                pipelineNameFP.write(",".join(pipelineResTitle) + "\n")
        
        
        for model in self.models:
            resultDict = dict()
            resultList = list()
            for sample in self.samples:
                res = sample.results[model.moduleName]
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

            # generate evaluation-used json result
            if (config.outputName is None):
                with open(f"{config.resultBase}/result-{pipelineIndex}.json", 'wt') as fp:
                    json.dump(resultDict, fp, indent=2)

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
            
            fileName = f'result-{pipelineIndex}.tsv' if config.outputName is None else config.outputName
            with open(f"{config.resultBase}/{fileName}", 'wt') as fp:
                fp.writelines(lines)

            if (isinstance(model, Pipeline) and config.outputName is None):
                params = model.getParamList()
                params.append(str(pipelineIndex))
                pipelineNameFP.write(",".join(params) + "\n")
                pipelineIndex += 1
            
        if (config.outputName is None):
            pipelineNameFP.close()
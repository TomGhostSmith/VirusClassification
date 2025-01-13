import os
import json
from Bio import SeqIO

from config import config
from entity.dataset import Dataset
from entity.sample import Sample

class ModelRunnder():
    def __init__(self, models, dataset:Dataset):
        def load(ids=None):
            queryFile = f"{config.datasetBase}/{config.datasetName}.fasta"
            answer = dict()
            if (os.path.exists(f"{config.datasetBase}/answer.json")):
                with open(f"{config.datasetBase}/answer.json") as fp:
                    answer = json.load(fp)

            sampleList = list()
            for record in SeqIO.parse(queryFile, "fasta"):
                if ids is None or record.id in ids:
                    sampleList.append(Sample(seq=record, stdResult=answer.get(record.id)))
            return sampleList
        
        self.models = models
        self.samples = list()

        config.majorDataset = dataset.majorDataset
        config.updatePath()
        ids = list()
        for minor in dataset.minorDatasets:
            with open(f"{config.datasetBase}/minorDatasets/{minor}") as fp:
                for line in fp:
                    ids.append(line.strip())
        ids = set(ids) if len(ids) > 0 else None
        self.samples += load(ids)

    def run(self):
        for model in self.models:
            model.getResults(self.samples)
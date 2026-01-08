import os
import json
import pandas

from prototype.module import Module
from entity.sample import Sample
from moduleResult.plainResult import PlainResult
from config import config
from utils import IOUtils

class DNATaxo(Module):
    def __init__(self, modelName, rank):
        super().__init__(f"DNA_taxo_{modelName}")
        self.name = modelName
        self.rank = rank

        cachedMapping = f"{config.cacheResultFolder}/DNA_mapping_{rank}.json"
        if (os.path.exists(cachedMapping)):
            with open(cachedMapping) as fp:
                id2Name = json.load(fp)
        else:
            level = self.rank.capitalize()
            taxamap_file = f'{config.modelRoot}/mapping/VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.json{level}_mapping.csv'
            taxamap_df = pandas.read_csv(taxamap_file)
            id2Name = {}
            for _, row in taxamap_df.iterrows():
                index = row[f"{self.rank.capitalize()} ID"]
                name = row[self.rank.capitalize()]
                if (index not in id2Name):
                    id2Name[index] = name
                elif (id2Name[index] != name):
                    IOUtils.showInfo(f"{self.rank} ID {index} corresponds to multiple names", "ERROR")
            with open(cachedMapping, 'wt') as fp:
                json.dump(id2Name, fp, indent=2)

        names = [None] * len(id2Name)
        for idx, name in id2Name.items():
            names[int(idx)] = name

        self.class_names = names

        self.cacheProbFile = f"{config.cacheResultFolder}/DNA_taxo_{modelName}_prob.tmp"
        self.cacheProbIndex = f"{config.cacheResultFolder}/DNA_taxo_{modelName}_prob.json"

        self.cachedSamples_prob = {"nextOffset": 0}
        self.nextOffset_prob = 0

    def run(self, samples:list[Sample]):
        if (os.path.exists(self.cacheProbIndex)):
            with open(self.cacheProbIndex) as fp:
                self.cachedSamples_prob = json.load(fp)
                self.nextOffset_prob = self.cachedSamples_prob["nextOffset"]

        fp = open(self.cacheProbFile)
        results = [self.getResult(sample, fp) for sample in samples]
        fp.close()

        return results

    def getResult(self, sample:Sample, fp):
        offset = self.cachedSamples_prob[sample.id]
        fp.seek(offset)
        line = fp.readline().strip('\n')
        probText = line[line.find('\t')+1:]
        probs = IOUtils.decodeBase64(probText)
        predictions = [PlainResult(pred, prob) for pred, prob in sorted(zip(self.class_names, probs), key=lambda x:x[1], reverse=True) if prob > 0]
        return predictions

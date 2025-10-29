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

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class ESMTaxo(Module):
    def __init__(self, modelName, maxLen, modelFolder, baseModelFolder, rank, pooling, batchSize=None):
        super().__init__(f"ESM_taxo_{modelName}_{pooling}")
        self.name = modelName
        if (pooling not in ["sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        self.pooling = pooling
        self.maxLen = maxLen
        self.modelFolder = modelFolder
        self.baseModelFolder = baseModelFolder
        
        self.batchSize = batchSize
        self.rank = rank

        cachedMapping = f"{config.cacheResultFolder}/ESM_mapping_{self.name}.json"
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
        
        # convert dict to list for better performance
        names = [None] * len(id2Name)
        for idx, name in id2Name.items():
            names[int(idx)] = name
        
        self.class_names = names

    def run(self, samples:list[Sample])->list[PlainResult]:
        NucleotideUtils.extractProtein(samples)
        proteinsToRun:list[ProteinSample] = list()
        for sample in samples:
            for protein in sample.proteins:
                protein.info[f"{self.rank}_labels"] = self.class_names
                proteinsToRun.append(protein)

        with ProcessPoolExecutor() as ex:
            results = ex.submit(getProbs, self.name, self.maxLen, self.modelFolder, self.baseModelFolder, len(self.class_names), self.batchSize, proteinsToRun).result()  # note: ex.submit() do not unpack params
            key = f"{self.name}_prob"
        
        for r, p in zip(results, proteinsToRun):
            p.info[key] = r


        results = [self.getResult(sample) for sample in samples]

        for p in proteinsToRun:
            p.info.pop(key)
        return results
    
    def getResult(self, sample:Sample):
        if (self.pooling == "sum"):
            votes = {n: 0 for n in self.class_names if "Unknown" not in n}
            for protein in sample.proteins:
                scores = protein.info[f"{self.name}_prob"]
                for taxo, score in zip(self.class_names, scores):
                    if ("Unknown" not in taxo):
                        votes[taxo] += score
        elif (self.pooling.startswith("top")):
            thresh = int(self.pooling[3:])
            votes = {}
            for protein in sample.proteins:
                scores = protein.info[f"{self.name}_prob"].tolist()
                rawScores = {taxo: score for taxo, score in zip(self.class_names, scores)}
                tops = sorted(list(rawScores.items()), key=lambda x:x[1], reverse=True)
                for taxo, score in tops[:thresh]:
                    if ("Unknown" not in taxo):
                        if (taxo in votes):
                            votes[taxo] += score
                        else:
                            votes[taxo] = score

        totalVotes = sum(votes.values())    # If pooling method == "sum", the totalVotes will be 1 * len(proteins) (not considering "Unknown" labels)
        if (len(votes) > 0 and totalVotes > 0):
            winner, maxVotes = max(votes.items(), key=lambda x:x[1])
            return PlainResult(winner, maxVotes/totalVotes)
        else:
            return None
        
def getProbs(name, maxLen, modelFolder, baseModelFolder, n_class, batchSize, proteinsToRun):
    model = ESMRunner(name, maxLen, modelFolder, baseModelFolder, n_class, batchSize)
    model.run(proteinsToRun, getProb=True)
    key = f"{name}_prob"
    results = [p.info[key] for p in proteinsToRun]
    return results
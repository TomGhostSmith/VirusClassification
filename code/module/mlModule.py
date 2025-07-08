# reconstructed
import os
import json
import pandas
from config import config
from prototype.module import Module
from moduleResult.mlResult import MLResult
from entity.sample import Sample
from module.esmRunner import ESMRunner
from tqdm import tqdm

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class MLModule(Module):
    def __init__(self, strategy="topdown", thresh=0.45, gen='1111000', pooling='sum'):
        self.strategy = strategy
        self.thresh = thresh
        self.gen = gen
        self.pooling = pooling
        if (pooling not in ["sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        super().__init__(f'ML-stratgy={strategy};th={thresh}, gen={gen}, pooling={pooling}')
        # self.baseName = self.moduleName
        self.resultDict:dict[str, MLResult] = dict()

        realmParams = [
            ("esm2_t33_256", 256, f"{config.modelRoot}/realm/esm2_t33_256"),
            ("esm2_t33_512", 512, f"{config.modelRoot}/realm/esm2_t33_512")
        ]
        kingdomParams = [
            ("esm2_t33_256", 256, f"{config.modelRoot}/kingdom/esm2_t33_256"),
            ("esm2_t33_512", 512, f"{config.modelRoot}/kingdom/esm2_t33_512")
        ]
        phylumParams = [
            ("esm2_t33_256", 256, f"{config.modelRoot}/phylum/esm2_t33_256"),
            ("esm2_t33_512", 512, f"{config.modelRoot}/phylum/esm2_t33_512")
        ]
        classParams = [
            ("esm2_t33_256", 256, f"{config.modelRoot}/class/esm2_t33_256"),
            ("esm2_t33_512", 512, f"{config.modelRoot}/class/esm2_t33_512")
        ]
        orderParams = [
            ("esm2_t33_512", 512, f"{config.modelRoot}/order/esm2_t33_512")
        ]
        familyParams = [
            ("esm2_t33_512", 512, f"{config.modelRoot}/family/esm2_t33_512"),
            ("esm2_t33_512_enlarge", 512, f"{config.modelRoot}/family/esm2_t33_512_enlarge")
        ]
        genusParams = [
            ("esm2_t33_256_enlarge", 256, f"{config.modelRoot}/genus/esm2_t33_256_enlarge_genus"),
            ("esm2_t33_256", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
        ]

        realmParam = realmParams[int(gen[0])]
        kingdomParam = kingdomParams[int(gen[1])]
        phylumParam = phylumParams[int(gen[2])]
        classParam = classParams[int(gen[3])]
        orderParam = orderParams[int(gen[4])]
        familyParam = familyParams[int(gen[5])]
        genusParam = genusParams[int(gen[6])]

        self.modelParams = {
            "realm": (*realmParam, "facebook/esm2_t33_650M_UR50D", 29, config.mlBatchSize),
            "kingdom": (*kingdomParam, "facebook/esm2_t33_650M_UR50D", 40, config.mlBatchSize),
            "phylum": (*phylumParam, "facebook/esm2_t33_650M_UR50D", 51, config.mlBatchSize),
            "class": (*classParam, "facebook/esm2_t33_650M_UR50D", 76, config.mlBatchSize),
            "order": (*orderParam, "facebook/esm2_t33_650M_UR50D", 981, config.mlBatchSize),
            "family": (*familyParam, "facebook/esm2_t33_650M_UR50D", 1129, config.mlBatchSize),
            "genus": (*genusParam, "facebook/esm2_t33_650M_UR50D", 3523, config.mlBatchSize),
        }

        
    def run(self, samples:list[Sample]):
        NucleotideUtils.extractProtein(samples)
        for sample in samples:
            self.resultDict[sample.id] = MLResult(self.strategy, self.thresh)

        unterminatedSamples = samples
        if (self.strategy.startswith('topdown')):
            for rank, param in self.modelParams.items():
                unterminatedSamples = self.runModel(unterminatedSamples, rank, param[0])
                if (len(unterminatedSamples) == 0):
                    break
        elif (self.strategy.startswith('bottomup')):
            for rank, param in reversed(list(self.modelParams.items())):
                unterminatedSamples = self.runModel(unterminatedSamples, rank, param[0])
                if (len(unterminatedSamples) == 0):
                    break
        else:  # highest, we need to run all the rank
            for rank, param in self.modelParams.items():
                self.runModel(samples, rank, param[0])

        results = list()
        for sample in samples:
            if (self.resultDict[sample.id].res is not None):
                results.append(self.resultDict[sample.id])
            else:
                results.append(None)
        
        return results


    def runModel(self, samples:list[Sample], rank:str, modelName:str)->list[Sample]:

        # abbr = self.modelParams[rank][1].split('/')[-1]
        cachedSamples:dict[str, int] = dict()
        cacheFile = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}.tmp"
        cacheIndex = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}.json"
        if (os.path.exists(cacheFile) and os.path.exists(cacheIndex)):
            with open(cacheIndex) as fp:
                cachedSamples = json.load(fp)
                nextOffset = cachedSamples["nextOffset"]
        else:
            nextOffset = 0
            cachedSamples["nextOffset"] = 0

        cachedMapping = f"{config.cacheResultFolder}/ESM_mapping_{rank}_{modelName}.json"
        if (os.path.exists(cachedMapping)):
            with open(cachedMapping) as fp:
                id2Name = json.load(fp)
        else:
            level = rank.capitalize()
            taxamap_file = f'{config.modelRoot}/mapping/VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.json{level}_mapping.csv'
            taxamap_df = pandas.read_csv(taxamap_file)
            id2Name = {}
            for _, row in taxamap_df.iterrows():
                index = row[f"{rank.capitalize()} ID"]
                name = row[rank.capitalize()]
                if (index not in id2Name):
                    id2Name[index] = name
                elif (id2Name[index] != name):
                    IOUtils.showInfo(f"{rank} ID {index} corresponds to multiple names", "ERROR")
            with open(cachedMapping, 'wt') as fp:
                json.dump(id2Name, fp, indent=2)
        
        # convert dict to list for better performance
        names = [None] * len(id2Name)
        for idx, name in id2Name.items():
            names[int(idx)] = name

        samplesToRun:list[Sample] = list()
        for sample in samples:
            if (sample.id not in cachedSamples):
                samplesToRun.append(sample)


        if (len(samplesToRun) > 0):
            model = ESMRunner(*self.modelParams[rank][1:])
            lines = model.run(samplesToRun)

            with open(cacheFile, 'at') as fp:
                if (nextOffset == 0):
                    line = "seq_name\t" + "\t".join(names) + "\n"
                    fp.write(line)
                    nextOffset += len(line)
                for seq_name, line in lines.items():
                    if (seq_name == "title"):
                        continue
                    cachedSamples[seq_name] = nextOffset
                    fp.write(line)
                    nextOffset += len(line)
                cachedSamples["nextOffset"] = nextOffset

            del model
            
            for sample in samplesToRun:
                if (sample.id not in cachedSamples):
                    cachedSamples[sample.id] = -1
            
            with open(cacheIndex, 'wt') as fp:
                json.dump(cachedSamples, fp, indent=2)
        

        unTerminatedSamples:list[Sample] = list()

        cachedResultFP = open(cacheFile)
        for sample in tqdm(samples, desc="pooling"):
            if (self.pooling == "sum"):
                votes = {n: 0 for n in names if "Unknown" not in n}
                for protein in sample.proteins:
                    offset = cachedSamples[protein.id]
                    if (offset == -1):
                        continue
                    cachedResultFP.seek(offset)
                    terms = cachedResultFP.readline().strip().split('\t')
                    scores = [float(t) for t in terms[1:]]
                    for taxo, score in zip(names, scores):
                        if ("Unknown" not in taxo):
                            votes[taxo] += score
            elif (self.pooling.startswith("top")):
                thresh = int(self.pooling[3:])
                votes = {}
                for protein in sample.proteins:
                    offset = cachedSamples[protein.id]
                    if (offset == -1):
                        continue
                    cachedResultFP.seek(offset)
                    terms = cachedResultFP.readline().strip().split('\t')
                    scores = [float(t) for t in terms[1:]]
                    rawScores = {taxo: score for taxo, score in zip(names, scores)}
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
                self.resultDict[sample.id].addResult(winner, maxVotes/totalVotes)
            
            if not (self.resultDict[sample.id].terminate):
                unTerminatedSamples.append(sample)

        cachedResultFP.close()
        
        return unTerminatedSamples
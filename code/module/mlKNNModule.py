# reconstructed
import os
import json
import pandas
from config import config
from prototype.module import Module
from moduleResult.mlResult import MLResult
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from module.esmRunner import ESMRunner
from tqdm import tqdm
import base64
import numpy

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class MLKNNModule(Module):
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
        cacheProbFile = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_prob.tmp"
        cacheCLSEmbFile = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_cls_emb.tmp"
        cacheAveEmbFile = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_ave_emb.tmp"
        cacheProbIndex = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_prob.json"
        cacheCLSEmbIndex = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_cls_emb.json"
        cacheAveEmbIndex = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_ave_emb.json"

        essentialFiles = [cacheProbFile, cacheCLSEmbFile, cacheAveEmbFile, cacheProbIndex, cacheCLSEmbIndex, cacheAveEmbIndex]
        allExists = True
        for f in essentialFiles:
            if (not os.path.exists(f)):
                allExists = False
        
        if (allExists):
            with open(cacheProbIndex) as fp:
                cachedSamples_prob = json.load(fp)
                nextOffset_prob = cachedSamples_prob["nextOffset"]
            with open(cacheCLSEmbIndex) as fp:
                cachedSamples_cls = json.load(fp)
                nextOffset_cls = cachedSamples_cls["nextOffset"]
            with open(cacheAveEmbIndex) as fp:
                cachedSamples_ave = json.load(fp)
                nextOffset_ave = cachedSamples_ave["nextOffset"]
        else:
            cachedSamples_prob = {"nextOffset": 0}
            nextOffset_prob = 0
            cachedSamples_cls = {"nextOffset": 0}
            nextOffset_cls = 0
            cachedSamples_ave = {"nextOffset": 0}
            nextOffset_ave = 0

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

        proteinsToRun:list[ProteinSample] = list()
        for sample in samples:
            for protein in sample.proteins:
                if (protein.id not in cachedSamples_prob or protein.id not in cachedSamples_cls or protein.id not in cachedSamples_ave):
                    proteinsToRun.append(protein)


        if (len(proteinsToRun) > 0):
            model = ESMRunner(*self.modelParams[rank][1:])
            lines = model.run(proteinsToRun)

            fp_prob = open(cacheProbFile, 'at')
            fp_cls = open(cacheCLSEmbFile, 'at')
            fp_ave = open(cacheAveEmbFile, 'at')
            

            # if (nextOffset_prob == 0):
            #     line = "seq_name\t" + "\t".join(names) + "\n"
            #     fp_prob.write(line)
            #     nextOffset_prob += len(line)
            for seq_name, (prob, cls, ave) in lines.items():
                cachedSamples_prob[seq_name] = nextOffset_prob
                cachedSamples_cls[seq_name] = nextOffset_cls
                cachedSamples_ave[seq_name] = nextOffset_ave

                probText = base64.b64encode(prob.tobytes()).decode('ascii')
                clsText = base64.b64encode(cls.tobytes()).decode('ascii')
                aveText = base64.b64encode(ave.tobytes()).decode('ascii')

                fp_prob.write(f"{seq_name}\t{probText}\n")
                fp_cls.write(f"{seq_name}\t{clsText}\n")
                fp_ave.write(f"{seq_name}\t{aveText}\n")
                nextOffset_prob += len(probText)
                nextOffset_cls += len(clsText)
                nextOffset_ave += len(aveText)

            cachedSamples_prob["nextOffset"] = nextOffset_prob
            cachedSamples_cls["nextOffset"] = nextOffset_cls
            cachedSamples_ave["nextOffset"] = nextOffset_ave

            fp_prob.close()
            fp_cls.close()
            fp_ave.close()

            del model
            
            for protein in proteinsToRun:
                if (protein.id not in cachedSamples_prob):
                    cachedSamples_prob[protein.id] = -1
                if (protein.id not in cachedSamples_cls):
                    cachedSamples_cls[protein.id] = -1
                if (protein.id not in cachedSamples_ave):
                    cachedSamples_ave[protein.id] = -1
            
            with open(cacheProbIndex, 'wt') as fp:
                json.dump(cachedSamples_prob, fp, indent=2)
            with open(cacheCLSEmbIndex, 'wt') as fp:
                json.dump(cachedSamples_cls, fp, indent=2)
            with open(cacheAveEmbIndex, 'wt') as fp:
                json.dump(cachedSamples_ave, fp, indent=2)
        

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
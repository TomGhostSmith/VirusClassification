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
            ("working/seq_name_genbank_2024_2024_exclusion.csv.1_2_5_10_30_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.SCL.genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
        ]

        self.modelParams = {}
        if (gen[0] != "N"):
            realmParam = realmParams[int(gen[0])]
            self.modelParams["realm"] = (*realmParam, "facebook/esm2_t33_650M_UR50D", 29, config.mlBatchSize)
        if (gen[1] != "N"):
            kingdomParam = kingdomParams[int(gen[1])]
            self.modelParams["kingdom"] = (*kingdomParam, "facebook/esm2_t33_650M_UR50D", 40, config.mlBatchSize)
        if (gen[2] != "N"):
            phylumParam = phylumParams[int(gen[2])]
            self.modelParams["phylum"] = (*phylumParam, "facebook/esm2_t33_650M_UR50D", 51, config.mlBatchSize)
        if (gen[3] != "N"):
            classParam = classParams[int(gen[3])]
            self.modelParams["class"] = (*classParam, "facebook/esm2_t33_650M_UR50D", 76, config.mlBatchSize)
        if (gen[4] != "N"):
            orderParam = orderParams[int(gen[4])]
            self.modelParams["order"] = (*orderParam, "facebook/esm2_t33_650M_UR50D", 981, config.mlBatchSize)
        if (gen[5] != "N"):
            familyParam = familyParams[int(gen[5])]
            self.modelParams["family"] = (*familyParam, "facebook/esm2_t33_650M_UR50D", 1129, config.mlBatchSize)
        if (gen[6] != "N"):
            genusParam = genusParams[int(gen[6])]
            self.modelParams["genus"] = (*genusParam, "facebook/esm2_t33_650M_UR50D", 3523, config.mlBatchSize)
        
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
        unTerminatedSamples:list[Sample] = list()

        if (os.path.exists(modelName)):
            df = pandas.read_csv(modelName)
            results = {}
            for _, row in df.iterrows():
                id = row["SampleID"]
                pred = row["Predicted_Genus_Name"]
                if ("Unknown" not in pred):
                    results[id] = pred
            for sample in samples:
                if (sample.id in results):
                    self.resultDict[sample.id].addResult(results[sample.id], 1)
                if not (self.resultDict[sample.id].terminate):
                    unTerminatedSamples.append(sample)
            return unTerminatedSamples

        cacheProbFile = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_prob.tmp"
        cacheProbIndex = f"{config.cacheResultFolder}/ESM_taxo_{rank}_{modelName}_prob.json"

        essentialFiles = [cacheProbFile, cacheProbIndex]
        allExists = True
        for f in essentialFiles:
            if (not os.path.exists(f)):
                allExists = False
        
        if (allExists):
            with open(cacheProbIndex) as fp:
                cachedSamples_prob = json.load(fp)
                nextOffset_prob = cachedSamples_prob["nextOffset"]
        else:
            cachedSamples_prob = {"nextOffset": 0}
            nextOffset_prob = 0

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
                if (protein.id not in cachedSamples_prob):
                    proteinsToRun.append(protein)


        if (len(proteinsToRun) > 0):
            IOUtils.showInfo(f"run {len(proteinsToRun)} proteins on ESM for {rank}")
            model = ESMRunner(*self.modelParams[rank][1:])
            lines = model.run(proteinsToRun)

            fp_prob = open(cacheProbFile, 'at')
            

            for seq_name, (prob, cls, ave) in lines.items():
                cachedSamples_prob[seq_name] = nextOffset_prob

                probText = f"{seq_name}\t{IOUtils.encodeBase64(prob)}\n"

                fp_prob.write(probText)
                nextOffset_prob += len(probText)

            cachedSamples_prob["nextOffset"] = nextOffset_prob

            fp_prob.close()

            del model
            
            for protein in proteinsToRun:
                if (protein.id not in cachedSamples_prob):
                    cachedSamples_prob[protein.id] = -1
            
            with open(cacheProbIndex, 'wt') as fp:
                json.dump(cachedSamples_prob, fp, indent=2)
        



        if (self.strategy in ["highest", "topdown", "bottomup"]):
            cachedResultFP_prob = open(cacheProbFile)
            for sample in tqdm(samples, desc=f"{rank} pooling"):
                if (self.pooling == "sum"):
                    votes = {n: 0 for n in names if "Unknown" not in n}
                    for protein in sample.proteins:
                        offset = cachedSamples_prob[protein.id]
                        if (offset == -1):
                            continue
                        cachedResultFP_prob.seek(offset)
                        line = cachedResultFP_prob.readline().strip()
                        scores = IOUtils.decodeBase64(line[line.find('\t')+1:])
                        for taxo, score in zip(names, scores):
                            if ("Unknown" not in taxo):
                                votes[taxo] += score
                elif (self.pooling.startswith("top")):
                    thresh = int(self.pooling[3:])
                    votes = {}
                    for protein in sample.proteins:
                        offset = cachedSamples_prob[protein.id]
                        if (offset == -1):
                            continue
                        cachedResultFP_prob.seek(offset)
                        line = cachedResultFP_prob.readline().strip()
                        scores = IOUtils.decodeBase64(line[line.find('\t')+1:])
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
            cachedResultFP_prob.close()

        # TODO: KNN and load trainset embedding
        elif (self.strategy == "ClsEmbKNN"):
            # cachedResultFP_cls = open(cacheCLSEmbFile)
            # cachedResultFP_cls.close()
            pass
        elif (self.strategy == "AveEmbKNN"):
            # cachedResultFP_ave = open(cacheAveEmbFile)
            # cachedResultFP_ave.close()
            pass

        
        return unTerminatedSamples
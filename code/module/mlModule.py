# reconstructed
import os
import json
import pandas
from concurrent.futures import ProcessPoolExecutor

from config import config
from prototype.module import Module
from moduleResult.mlResult import MLResult
from moduleResult.plainResult import PlainResult
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from module.esmTaxo import ESMTaxo
from tqdm import tqdm

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class MLModule(Module):
    def __init__(self, strategy="topdown", thresh=0.45, gen='1111000', pooling='sum'):
        self.strategy = strategy
        self.thresh = thresh
        self.gen = gen
        self.pooling = pooling
        if (strategy not in ["highest", "topdown", "bottomup"]):
            raise ValueError("Unsupported strategy")
        if (pooling not in ["sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        super().__init__(f'ML-stratgy={strategy};th={thresh}, gen={gen}, pooling={pooling}')
        # self.baseName = self.moduleName
        self.resultDict:dict[str, MLResult] = dict()

        realmParams = [
            ("realm_esm2_t33_256", 256, f"{config.modelRoot}/realm/esm2_t33_256"),
            ("realm_esm2_t33_512", 512, f"{config.modelRoot}/realm/esm2_t33_512")
        ]
        kingdomParams = [
            ("kingdom_esm2_t33_256", 256, f"{config.modelRoot}/kingdom/esm2_t33_256"),
            ("kingdom_esm2_t33_512", 512, f"{config.modelRoot}/kingdom/esm2_t33_512")
        ]
        phylumParams = [
            ("phylum_esm2_t33_256", 256, f"{config.modelRoot}/phylum/esm2_t33_256"),
            ("phylum_esm2_t33_512", 512, f"{config.modelRoot}/phylum/esm2_t33_512")
        ]
        classParams = [
            ("class_esm2_t33_256", 256, f"{config.modelRoot}/class/esm2_t33_256"),
            ("class_esm2_t33_512", 512, f"{config.modelRoot}/class/esm2_t33_512")
        ]
        orderParams = [
            ("order_esm2_t33_512", 512, f"{config.modelRoot}/order/esm2_t33_512")
        ]
        familyParams = [
            ("family_esm2_t33_512", 512, f"{config.modelRoot}/family/esm2_t33_512"),
            ("family_esm2_t33_512_enlarge", 512, f"{config.modelRoot}/family/esm2_t33_512_enlarge")
        ]
        genusParams = [
            ("genus_esm2_t33_256_enlarge", 256, f"{config.modelRoot}/genus/esm2_t33_256_enlarge_genus"),
            ("genus_esm2_t33_256", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.1_2_5_10_30_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.SCL.genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.1_2_3_4_5_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.len_1_lr_1e-5_hidden_dim_512_SCL_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.len_1_SCL_temp_0.05_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.len_1_SCL_hiddendim_512_enlarge_protein_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.len_1_SCL_enlarge_protein_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.len_1_SCL_temp_0.05_best_enlarge_protein_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.len_1_SCL_temp_0.15_best_enlarge_protein_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.CNN_online_1to10_bs2048_kernel_1_3_5_num_filters_512_lr_1e-4_weight_decay_1e-4_best_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("working/seq_name_genbank_2024_2024_exclusion.csv.CNN_online_1to10_bs2048_kernel_1_3_5_num_filters_512_lr_1e-4_weight_decay_1e-4_enlarge_best_genus_predictions.csv", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
        ]

        self.modelParams = {}
        if (gen[0] != "N"):
            realmParam = realmParams[ord(gen[0]) - 48]  # 48 is the ascii of "0"
            self.modelParams["realm"] = (*realmParam, "facebook/esm2_t33_650M_UR50D", 29)
        if (gen[1] != "N"):
            kingdomParam = kingdomParams[ord(gen[1]) - 48]
            self.modelParams["kingdom"] = (*kingdomParam, "facebook/esm2_t33_650M_UR50D", 40)
        if (gen[2] != "N"):
            phylumParam = phylumParams[ord(gen[2]) - 48]
            self.modelParams["phylum"] = (*phylumParam, "facebook/esm2_t33_650M_UR50D", 51)
        if (gen[3] != "N"):
            classParam = classParams[ord(gen[3]) - 48]
            self.modelParams["class"] = (*classParam, "facebook/esm2_t33_650M_UR50D", 76)
        if (gen[4] != "N"):
            orderParam = orderParams[ord(gen[4]) - 48]
            self.modelParams["order"] = (*orderParam, "facebook/esm2_t33_650M_UR50D", 981)
        if (gen[5] != "N"):
            familyParam = familyParams[ord(gen[5]) - 48]
            self.modelParams["family"] = (*familyParam, "facebook/esm2_t33_650M_UR50D", 1129)
        if (gen[6] != "N"):
            genusParam = genusParams[ord(gen[6]) - 48]
            self.modelParams["genus"] = (*genusParam, "facebook/esm2_t33_650M_UR50D", 3523)
        
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
        
        with ProcessPoolExecutor() as ex:
            results = ex.submit(runESM, self.modelParams[rank][:-1], samples, rank, self.pooling).result()  # note: ex.submit() do not unpack params
        
        for sample in samples:
            taxo, score = results[sample.id]
            if (taxo is not None and score is not None):
                self.resultDict[sample.id].addResult(taxo, score)

            if not (self.resultDict[sample.id].terminate):
                unTerminatedSamples.append(sample)

        return unTerminatedSamples
    
def runESM(params, samples:list[Sample], rank, pooling):
    model = ESMTaxo(*params, rank=rank, pooling=pooling)  # currently we do not need to pass n_class
    mName = model.moduleName
    model.getResults(samples)
    results = {}
    
    for sample in samples:
        res:PlainResult = sample.results[mName]
        if (res):
            results[sample.id] = (res.pred, res.score)
        else:
            results[sample.id] = (None, None)

    return results
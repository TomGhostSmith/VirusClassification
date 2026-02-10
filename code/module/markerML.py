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
from module.marker import Marker
from entity.taxoTree import taxoTree

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class MarkerML(Module):
    def __init__(self, trainset, strategy="topdown", thresh=0.45, gen='1111000'):
        self.strategy = strategy
        self.trainset = trainset
        self.thresh = thresh
        self.gen = gen
        if (strategy not in ["highest", "topdown", "bottomup"]):
            raise ValueError("Unsupported strategy")
        super().__init__(f'markerML-train-{trainset};stratgy={strategy};th={thresh}, gen={gen}')
        # self.baseName = self.moduleName
        self.resultDict:dict[str, MLResult] = dict()

        realmParams = [
            ("realm_esm2_t33_256", 256, f"{config.modelRoot}/realm/esm2_t33_256"),
            ("realm_esm2_t33_512", 512, f"{config.modelRoot}/realm/esm2_t33_512"),
            ("realm_esm2_t33_1022", 1022, f"{config.modelRoot}/realm/esm2_t33_1022")
        ]
        kingdomParams = [
            ("kingdom_esm2_t33_256", 256, f"{config.modelRoot}/kingdom/esm2_t33_256"),
            ("kingdom_esm2_t33_512", 512, f"{config.modelRoot}/kingdom/esm2_t33_512"),
            ("kingdom_esm2_t33_1022", 1022, f"{config.modelRoot}/kingdom/esm2_t33_1022")
        ]
        phylumParams = [
            ("phylum_esm2_t33_256", 256, f"{config.modelRoot}/phylum/esm2_t33_256"),
            ("phylum_esm2_t33_512", 512, f"{config.modelRoot}/phylum/esm2_t33_512"),
            ("phylum_esm2_t33_1022", 1022, f"{config.modelRoot}/phylum/esm2_t33_1022")
        ]
        classParams = [
            ("class_esm2_t33_256", 256, f"{config.modelRoot}/class/esm2_t33_256"),
            ("class_esm2_t33_512", 512, f"{config.modelRoot}/class/esm2_t33_512"),
            ("class_esm2_t33_1022", 1022, f"{config.modelRoot}/class/esm2_t33_1022")
        ]
        orderParams = [
            ("order_esm2_t33_512", 512, f"{config.modelRoot}/order/esm2_t33_512"),
            ("order_esm2_t33_1022", 1022, f"{config.modelRoot}/order/esm2_t33_1022")
        ]
        familyParams = [
            ("family_esm2_t33_512", 512, f"{config.modelRoot}/family/esm2_t33_512"),
            ("family_esm2_t33_512_enlarge", 512, f"{config.modelRoot}/family/esm2_t33_512_enlarge"),
            ("family_esm2_t33_1022", 1022, f"{config.modelRoot}/family/esm2_t33_1022")
        ]
        genusParams = [
            ("genus_esm2_t33_256_enlarge", 256, f"{config.modelRoot}/genus/esm2_t33_256_enlarge_genus"),
            ("genus_esm2_t33_256", 256, f"{config.modelRoot}/genus/esm2_t33_256_order_family_finetune"),
            ("genus_esm2_t33_1022", 1022, f"{config.modelRoot}/genus/esm2_t33_650M_UR50D_MAXLEN_1022_bs2x4_accum2_lr3e-5_ep10"),
            ("genus_esm2_t33_1022_2", 1022, f"{config.modelRoot}/genus/esm2_t33_650M_UR50D_MAX_LENGTH_1022_per_device_batch_size_2_num_train_epochs_10_save_steps_10000_lr_3e-5_save_steps_10000"),
            ("genus_esm2_t33_1022_enlarge", 1022, f"{config.modelRoot}/genus/enlarge_esm2_t33_650M_UR50D_MAX_LENGTH_1022_per_device_batch_size_2_num_train_epochs_10_save_steps_10000_lr_3e-5_save_steps_10000"),
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
        
    def run(self, samples:list[Sample], **kwargs):
        NucleotideUtils.extractProtein(samples)
        for sample in samples:
            self.resultDict[sample.id] = MLResult(self.strategy, 0)

        markerModule = Marker(self.trainset, "sum", coverage=80, identity=50)
        markerModule.run(samples, keepVotes=True)

        for sample in samples:
            newVotes = {r: {} for r in config.resultRanks}
            for name, vote in sample.info[f"{markerModule.moduleName}_votes"].items():
                for n in taxoTree.ICTVTree.nodes[name].path:
                    if n.rank in newVotes:
                        if (n.name not in newVotes[n.rank]):
                            newVotes[n.rank][n.name] = vote
                        else:
                            newVotes[n.rank][n.name] += vote

            # normalize marker votes
            for rank, rankVotes in newVotes.items():
                total = sum(rankVotes.values())
                for k, v in rankVotes.items():
                    rankVotes[k] = v/total

            sample.info["votes"] = newVotes
            sample.info.pop(f"{markerModule.moduleName}_votes")

        unterminatedSamples = samples
        if (self.strategy.startswith('topdown')):
            for rank, param in self.modelParams.items():
                unterminatedSamples = self.runModel(unterminatedSamples, rank)
                if (len(unterminatedSamples) == 0):
                    break
        elif (self.strategy.startswith('bottomup')):
            for rank, param in reversed(list(self.modelParams.items())):
                unterminatedSamples = self.runModel(unterminatedSamples, rank)
                if (len(unterminatedSamples) == 0):
                    break
        else:  # highest, we need to run all the rank
            for rank, param in self.modelParams.items():
                self.runModel(samples, rank)

        results = list()
        for sample in samples:
            if (self.resultDict[sample.id].res is not None):
                results.append([self.resultDict[sample.id]])
            else:
                results.append(None)
        
        for sample in samples:
            sample.info.pop("votes")
        
        return results


    def runModel(self, samples:list[Sample], rank:str)->list[Sample]:
        unTerminatedSamples:list[Sample] = list()
        
        model = ESMTaxo(*self.modelParams[rank][:-1], rank=rank, pooling="sum")  # currently we do not need to pass n_class
        mName = model.moduleName
        model.run(samples, keepVotes=True)
        
        for sample in samples:
            if (len(sample.proteins) == 0):
                continue
            votes = sample.info[f"votes"][rank]
            mlVotes = sample.info[f"{mName}_votes"]
            # normalize ml votes and add them to the normalized marker votes
            for k, v in mlVotes.items():
                if k in votes:
                    votes[k] += v/len(sample.proteins)
                else:
                    votes[k] = v/len(sample.proteins)

            if (self.thresh < 1):
                if (len(votes) > 0):
                    winner, maxVotes = max(votes.items(), key=lambda x: x[1])
                    if (maxVotes > self.thresh * 2):
                        self.resultDict[sample.id].addResult(winner, maxVotes / 2)
            else:
                if (len(votes) > 1):
                    (top1Name, top1Votes), (top2Name, top2Votes) = sorted(list(votes.items()), key=lambda x:x[1], reverse=True)[:2]
                    if (top1Votes > top2Votes * self.thresh):
                        self.resultDict[sample.id].addResult(top1Name, top1Votes / 2)

            if not (self.resultDict[sample.id].terminate):
                unTerminatedSamples.append(sample)

        return unTerminatedSamples
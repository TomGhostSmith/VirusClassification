# reconstructed
import os
import json
import pandas

from config import config
from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult
from module.esmRunner import ESMRunner
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from utils.NucleotideUtils import NucleotideUtils

class ESMIdentify(Module):
    def __init__(self):
        self.viruses = set()
        # names = {
        #     "150M_256": "esm2_t30_150M_UR50D_dnainput_scl_MAX_LENGTH_256_predicted_virus_names.tsv",
        #     "150M_512": "esm2_t30_150M_UR50D_MAX_LENGTH_512_predicted_virus_names.tsv",
        #     "650M_256": "esm2_t33_650M_UR50D_MAX_LENGTH_256_predicted_virus.tsv",
        #     "650M_256_merge": "esm2_t33_650M_UR50D_MAX_LENGTH_256_12_result.csv.merge_gene_to_contig.csv.phage.tsv"
        # }
        # self.models = {
        #     "150M_512": f"{config.modelRoot}/viral_identify/esm2_t30_512"
        # }
        # self.fullName = names[shortName]
        # self.modelPath = self.models[shortName]
        
        super().__init__(f'esm-150M_512')

    def run(self, samples:list[Sample], **kwargs):
        NucleotideUtils.extractProtein(samples)
        results = list()

        proteinsToRun:list[ProteinSample] = list()
        for sample in samples:
            for protein in sample.proteins:
                proteinsToRun.append(protein)
        
        model = ESMRunner("identify", 512, f"{config.modelRoot}/viral_identify/esm2_t30_512", "facebook/esm2_t30_150M_UR50D", 2, config.esmBatchSize)
        model.run(proteinsToRun, getProb=True)
        del model

        key = "identify_prob"

        for sample in samples:
            totalScore = 0
            validProteinCount = 0
            for protein in sample.proteins:
                if (key in protein.info):
                    totalScore += protein.info[key][1]
                    validProteinCount += 1
            
            if (validProteinCount > 0):
                s = totalScore / validProteinCount
                results.append([VirusPredictionResult(s, s>=0.5)])
            else:
                results.append(None)

        for sample in samples:
            for protein in sample.proteins:
                protein.info.pop(key, None)
        
        return results
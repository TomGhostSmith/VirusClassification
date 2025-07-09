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

class ESM(Module):
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

    def run(self, samples:list[Sample]):
        results = list()

        cacheFile = f"{config.cacheResultFolder}/ESM_pred.json"
        cachedSamples:dict[str, float] = {}  # the score of being a virus
        if (os.path.exists(cacheFile)):
            with open(cacheFile) as fp:
                cachedSamples = json.load(fp)

        proteinsToRun:list[ProteinSample] = list()
        for sample in samples:
            for protein in sample.proteins:
                if protein.id not in cachedSamples:
                    proteinsToRun.append(sample)
        
        if (len(proteinsToRun) > 0):
            self.model = ESMRunner(512, f"{config.modelRoot}/viral_identify/esm2_t30_512", "facebook/esm2_t30_150M_UR50D", 2, config.esmBatchSize)
            lines = self.model.run(samples)

            for seqName, line in lines.items():
                terms = line.strip().split('\t')
                cachedSamples[seqName] = float(terms[2])
            
            del self.model

            for protein in proteinsToRun:
                if (protein.id not in cachedSamples):
                    cachedSamples[protein.id] = 'N/A'

            with open(cacheFile, 'wt') as fp:
                json.dump(cachedSamples, fp, indent=2)

        for sample in samples:
            totalScore = 0
            validProteinCount = 0
            for protein in sample.proteins:
                if (cachedSamples[protein.id] != 'N/A'):
                    totalScore += cachedSamples[protein.id]
                    validProteinCount += 1
            
            if (validProteinCount > 0):
                if (totalScore/validProteinCount > 0.5):
                    results.append(VirusPredictionResult())
                else:
                    results.append(None)
            else:
                results.append(None)
        
        return results
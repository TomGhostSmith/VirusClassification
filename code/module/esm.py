from config import config
from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult

class ESM(Module):
    def __init__(self, shortName):
        self.viruses = set()
        names = {
            "150M_256": "esm2_t30_150M_UR50D_dnainput_scl_MAX_LENGTH_256_predicted_virus_names.tsv",
            "150M_512": "esm2_t30_150M_UR50D_MAX_LENGTH_512_predicted_virus_names.tsv",
            "650M_256": "esm2_t33_650M_UR50D_MAX_LENGTH_256_predicted_virus.tsv"
        }
        self.fullName = names[shortName]
        super().__init__(f'esm-{shortName}')

    def run(self):
        with open(f'{config.virusPredResultFolder}/{self.fullName}') as fp:
            self.viruses = {l.strip() for l in fp}
    
    def getResult(self, sample):
        if (sample.query in self.viruses):
            return VirusPredictionResult()
        else:
            return None
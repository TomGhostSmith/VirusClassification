from config import config
from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult

class ESM(Module):
    def __init__(self):
        self.moduleName = 'esm'
        self.viruses = set()

    def run(self):
        with open(f'{config.virusPredResultFolder}/esm2_t30_150M_UR50D_MAX_LENGTH_512_predicted_virus_names.tsv') as fp:
            self.viruses = {l.strip() for l in fp}
    
    def getResult(self, sample):
        if (sample.query in self.viruses):
            return VirusPredictionResult()
        else:
            return None
from prototype.result import Result
from entity.taxoTree import taxoTree

class VirusPredictionResult(Result):
    def __init__(self, score=1, isVirus=True):
        super().__init__()
        self.score = score
        self.isVirus = isVirus
    
    def calcTaxoNode(self):
        if (self.node is None):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName='Viruses')
from prototype.result import Result
from entity.taxoTree import taxoTree

class VirusPredictionResult(Result):
    def __init__(self):
        super().__init__()
    
    def calcTaxoNode(self):
        if (self.node is None):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName='Viruses')
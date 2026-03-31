from prototype.result import Result

class VirusPredictionResult(Result):
    def __init__(self, score=1, isVirus=True):
        super().__init__()
        self.score = score
        self.isVirus = isVirus
    
    def calcTaxoNode(self):
        if (self.node is None):
            from entity.taxoTree import taxoTree
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName='Viruses')
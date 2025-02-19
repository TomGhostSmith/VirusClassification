from prototype.result import Result
from entity.taxoTree import taxoTree

class PhaGCNResult(Result):
    def __init__(self, pred:str):
        super().__init__()
        if (pred.endswith('_like')):
            self.pred = pred[:-5]
            self.isVague = True
        else:
            self.pred = pred
            self.isVague = False

    def calcTaxoNode(self):
        self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.pred)
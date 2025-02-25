from prototype.result import Result
from entity.taxoTree import taxoTree

class PlainResult(Result):
    def __init__(self, pred:str):
        super().__init__()
        if (pred.endswith('_like')):
            self.pred = pred[:-5]
            self.isVague = True
        else:
            self.pred = pred
            self.isVague = False

    def calcTaxoNode(self):
        if (self.pred in taxoTree.ICTVTree.name2ID):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.pred)
        elif (self.pred in taxoTree.viralNCBITree.name2ID):
            self.node = taxoTree.getTaxoNodeFromNCBI(NCBIName=self.pred)
        else:
            self.node = None
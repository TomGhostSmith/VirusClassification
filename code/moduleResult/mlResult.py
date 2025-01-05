from prototype.result import Result
from entity.taxoTree import taxoTree

class MLResult(Result):
    def __init__(self, name):
        super().__init__()
        self.name = name   # the name should be a ICTV name

    def calcTaxoNode(self):
        if (self.node is None):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.name)
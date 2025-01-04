from prototype.result import Result
from entity.taxoTree import taxoTree

class MLResult(Result):
    def __init__(self, name):
        self.name = name   # the name should be a ICTV name

    def toTaxoNode(self):
        return taxoTree.getTaxoNodeFromICTV(ICTVName=self.name)
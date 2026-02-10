from prototype.result import Result
from moduleResult.diamondAlignment import DiamondAlignment
from entity.taxoTree import taxoTree
from config import config

class MarkerResult(Result):
    def __init__(self, alignment:DiamondAlignment, markerName:str):
        super().__init__()
        self.alignment = alignment
        self.markerName = markerName
        self.scores = dict()
        
    def calcTaxoNode(self):
        if (self.node is None):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.markerName)
            
            score = self.alignment.similarity
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = score
            self.score = score
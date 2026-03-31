from prototype.result import Result
from moduleResult.alignment import Alignment

class MinimapResult(Result):
    def __init__(self, alignment:Alignment):
        super().__init__()
        self.alignment = alignment
        self.rank = 'species'
        self.scores = dict()

    def setTargetRank(self, rank):
        self.rank = rank
        self.node = None
    
    # only return the taxoNode for the best alignment
    def calcTaxoNode(self):
        if (self.node is None):
            from entity.taxoTree import taxoTree
            from config import config
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromAccession(self.alignment.ref)
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            score = 1 - 10 ** (-self.alignment.quality/10)
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = score
            self.score = score
    
    def __copy__(self):
        obj = MinimapResult()
        obj.alignment = self.alignment          # shallow copy, alignments are read only
        obj.rank = self.rank                    # a string, which will generate a new object
        obj.scores = self.scores.copy()         # deep copy the scores inside the list
        return obj
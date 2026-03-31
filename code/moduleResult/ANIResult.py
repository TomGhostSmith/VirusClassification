from prototype.result import Result
from moduleResult.ANIAlignment import ANIAlignment
from config import config

class ANIResult(Result):
    def __init__(self, alignment:ANIAlignment):
        super().__init__()
        self.alignment = alignment
        self.rank = 'species'
        self.scores = dict()
    
    # only return the taxoNode for the best alignment
    def calcTaxoNode(self):
        if (self.node is None):
            from entity.taxoTree import taxoTree
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromAccession(self.alignment.ref)
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = self.alignment.overallIdentity

            self.score = self.alignment.overallIdentity
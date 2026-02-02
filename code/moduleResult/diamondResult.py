from prototype.result import Result
from moduleResult.diamondAlignment import DiamondAlignment
from entity.taxoTree import taxoTree
from config import config

class DiamondResult(Result):
    def __init__(self, alignment:DiamondAlignment):
        super().__init__()
        self.alignment = alignment
        self.rank = 'species'
        self.scores = dict()

    def setTargetRank(self, rank):
        self.rank = rank
        
    def calcTaxoNode(self):
        if (self.node is None):
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromAccession(self.alignment.refContig)
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            score = self.alignment.similarity / 100
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = score
            self.score = score

        elif (config.rankLevels[self.node.ICTVNode.rank] > config.rankLevels[self.rank]):
            targetRankLevel = config.rankLevels[self.rank]
            for n in reversed(self.node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
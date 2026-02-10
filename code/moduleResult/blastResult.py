from prototype.result import Result
from moduleResult.blastAlignment import BlastAlignment
from entity.taxoTree import taxoTree
from config import config

class BlastResult(Result):
    def __init__(self, alignment:BlastAlignment):
        super().__init__()
        self.alignment:BlastAlignment = alignment
        self.rank = 'species'
        self.scores = dict()

    def setTargetRank(self, rank):
        self.rank = rank
        
    def calcTaxoNode(self):
        if (self.node is None):
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromAccession(self.alignment.ref)
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            score = self.alignment.similarity
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = score
            self.score = score

        elif (config.rankLevels[self.node.ICTVNode.rank] > config.rankLevels[self.rank]):
            targetRankLevel = config.rankLevels[self.rank]
            for n in reversed(self.node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
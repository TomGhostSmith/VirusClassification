from prototype.result import Result
from moduleResult.blastAlignment import BlastAlignment
from entity.taxoTree import taxoTree
from config import config

class BlastResult(Result):
    def __init__(self):
        super().__init__()
        self.alignments:list[BlastAlignment] = list()
        self.bestAlignment:BlastAlignment = None
        self.rank = 'species'
        self.scores = dict()
    
    def addAlignment(self, alignment:BlastAlignment):
        self.alignments.append(alignment)
        if (self.bestAlignment is None):
            self.bestAlignment = alignment
        elif (alignment.betterThan(self.bestAlignment)):
            self.bestAlignment = alignment

    def setTargetRank(self, rank):
        self.rank = rank
        
    def calcTaxoNode(self):
        if (self.node is None):
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromICTV(ICTVID=self.bestAlignment.ref)
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            score = 1
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = score

        elif (config.rankLevels[self.node.ICTVNode.rank] > config.rankLevels[self.rank]):
            targetRankLevel = config.rankLevels[self.rank]
            for n in reversed(self.node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
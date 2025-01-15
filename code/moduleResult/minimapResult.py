from prototype.result import Result
from moduleResult.alignment import Alignment
from entity.taxoTree import taxoTree
from config import config

class MinimapResult(Result):
    def __init__(self):
        super().__init__()
        self.alignments = list()
        self.bestAlignment = None
        self.rank = 'species'
        self.scores = dict()

    def addAlignment(self, alignment:Alignment):
        self.alignments.append(alignment)
        if (self.bestAlignment is None):
            self.bestAlignment = alignment
        elif (alignment.betterThan(self.bestAlignment)):
            self.bestAlignment = alignment

    def setTargetRank(self, rank):
        self.rank = rank
    
    # only return the taxoNode for the best alignment
    def calcTaxoNode(self):
        if (self.node is None):
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromICTV(ICTVID=self.bestAlignment.ref)
            align60Counts = 0
            for alignment in self.alignments:
                if alignment.quality == 60:
                    align60Counts += 1
            # if (align60Counts > 0):
            #     self.score = 1/align60Counts
            # else:
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            score = 1 - 10 ** (-self.bestAlignment.quality/10)
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = score

        elif (config.rankLevels[self.node.ICTVNode.rank] > config.rankLevels[self.rank]):
            targetRankLevel = config.rankLevels[self.rank]
            for n in reversed(self.node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
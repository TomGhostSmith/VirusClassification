from prototype.result import Result
from moduleResult.ANIAlignment import ANIAlignment
from entity.taxoTree import taxoTree
from config import config

class ANIResult(Result):
    def __init__(self, alignment:ANIAlignment):
        super().__init__()
        self.alignment = alignment
        self.rank = 'species'
        self.scores = dict()

    def setTargetRank(self, rank):
        self.rank = rank
    
    # only return the taxoNode for the best alignment
    def calcTaxoNode(self):
        if (self.node is None):
            targetRankLevel = config.rankLevels[self.rank]
            node = taxoTree.getTaxoNodeFromAccession(self.alignment.ref)
            # align60Counts = 0
            # for alignment in self.alignments:
            #     if alignment.quality == 60:
            #         align60Counts += 1
            # if (align60Counts > 0):
            #     self.score = 1/align60Counts
            # else:
                
            for n in reversed(node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
            
            # score = 1 - 10 ** (-self.alignment.quality/10)
            for n in self.node.ICTVNode.path:
                self.scores[n.rank] = self.alignment.overallIdentity

            self.score = self.alignment.overallIdentity

        elif (config.rankLevels[self.node.ICTVNode.rank] > config.rankLevels[self.rank]):
            targetRankLevel = config.rankLevels[self.rank]
            for n in reversed(self.node.ICTVNode.path):
                if config.rankLevels[n.rank] <= targetRankLevel:
                    self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=n)
                    break
    
    def __copy__(self):
        obj = ANIResult()
        obj.alignment = self.alignment  # shallow copy, alignments are read only
        obj.rank = self.rank                    # a string, which will generate a new object
        obj.scores = self.scores.copy()         # deep copy the scores inside the list
        return obj
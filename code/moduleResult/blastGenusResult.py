from prototype.result import Result
from moduleResult.blastAlignment import BlastAlignment
from entity.taxoTree import taxoTree
from config import config

class BlastGenusResult(Result):
    def __init__(self, genus):
        super().__init__()
        self.genus = genus
        self.alignments:list[BlastAlignment] = []
        self.score = 0

    def addAlignment(self, alignment:BlastAlignment):
        self.alignments.append(alignment)
        if alignment.similarity > self.score:
            self.score = alignment.similarity
        
    def calcTaxoNode(self):
        if (self.node is None):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.genus)
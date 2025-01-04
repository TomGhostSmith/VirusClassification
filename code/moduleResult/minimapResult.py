from prototype.result import Result
from moduleResult.alignment import Alignment
from entity.taxoTree import taxoTree

class MinimapResult(Result):
    def __init__(self):
        self.alignments = list()
        self.bestAlignment = None

    def addAlignment(self, alignment:Alignment):
        self.alignments.append(alignment)
        if (self.bestAlignment is None):
            self.bestAlignment = alignment
        elif (alignment.betterThan(self.bestAlignment)):
            self.bestAlignment = alignment
    
    # only return the taxoNode for the best alignment
    def toTaxoNode(self):
        return taxoTree.getTaxoNodeFromICTV(ICTVID=self.bestAlignment.ref)
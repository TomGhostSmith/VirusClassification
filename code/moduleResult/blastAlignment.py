# reconstructing
import re
from entity.taxoTree import taxoTree


class BlastAlignment():
    # def __init__(self, ref, quality, cigar):
    def __init__(self, alignment:str):
        terms = alignment.strip().split('\t')

        self.ref = taxoTree.ICTVTree.accession2ID[terms[1]]
        self.similarity = float(terms[2])
        self.length = int(terms[3])
        self.mismatch = int(terms[4])
        self.gapopen = int(terms[5])
        self.qstart = int(terms[6])
        self.qend = int(terms[7])
        self.refstart = int(terms[8])
        self.refend = int(terms[9])
        self.evalue = float(terms[10])
        self.bitscore = float(terms[11])

        self.queryCoverLength = self.qend - self.qstart
        self.refCoverLength = self.refend - self.refstart
        

    def betterThan(self, alignment):
        if (self.similarity > alignment.similarity):
            return True
        else:
            return False
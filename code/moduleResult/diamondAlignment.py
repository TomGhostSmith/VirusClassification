# reconstructing
import re
from entity.taxoTree import taxoTree


class DiamondAlignment():
    # def __init__(self, ref, quality, cigar):
    def __init__(self, alignment:str):
        terms = alignment.strip().split('\t')

        self.ref = terms[1]
        self.refContig = terms[1].rsplit('_', 1)[0]
        self.similarity = float(terms[2])  # note: diamond(blastp) 0-100, mmseqs 0-1
        self.length = int(terms[3])
        self.mismatch = int(terms[4])
        self.gapopen = int(terms[5])
        self.qstart = int(terms[6])
        self.qend = int(terms[7])
        self.refstart = int(terms[8])
        self.refend = int(terms[9])
        self.evalue = float(terms[10])
        self.bitscore = float(terms[11])
        self.likelihood = 0

        self.queryCoverLength = self.qend - self.qstart
        self.refCoverLength = self.refend - self.refstart
        

    def betterThan(self, alignment):
        if (self.similarity > alignment.similarity):
            return True
        else:
            return False
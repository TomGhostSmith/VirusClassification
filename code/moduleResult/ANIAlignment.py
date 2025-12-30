# reconstructing
import re


class ANIAlignment():
    # def __init__(self, ref, quality, cigar):
    def __init__(self, alignment:str):
        terms = alignment.strip().split('\t')

        self.ref = terms[1]
        self.identity = float(terms[2])
        self.matches = int(terms[3])
        self.fragments = int(terms[4])

        self.matchRatio = self.matches / self.fragments
        self.overallIdentity = self.identity * self.matchRatio        

    def betterThan(self, alignment):
        if (self.overallIdentity > alignment.overallIdentity):
            return True
        else:
            return False
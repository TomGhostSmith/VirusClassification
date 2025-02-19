from prototype.result import Result
from entity.taxoTree import taxoTree
from config import config

class GenomadResult(Result):
    def __init__(self, line:str):
        super().__init__()
        self.finalSpecies = None
        terms = line.strip().split('\t')
        taxoPath = terms[10].split(';')
        self.taxoPath = list()
        for taxoName in taxoPath:
            if taxoName != "":
                self.taxoPath.append(taxoName)
        self.finalSpecies = self.taxoPath[-1]
        self.geneCount = int(terms[4])
        self.virusScore = float(terms[6])
        self.hallmarks = int(terms[8])
        self.markerEnrichment = float(terms[9])

    def calcTaxoNode(self):
        self.node = taxoTree.getTaxoNodeFromNCBI(NCBIName=self.finalSpecies)
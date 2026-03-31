from prototype.result import Result

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
        self.score = self.virusScore

    def calcTaxoNode(self):
        from entity.taxoTree import taxoTree
        if (self.finalSpecies in taxoTree.viralNCBITree.name2ID):
            self.node = taxoTree.getTaxoNodeFromNCBI(NCBIName=self.finalSpecies)
        elif (self.finalSpecies in taxoTree.ICTVTree.name2ID):
            self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.finalSpecies)
        else:
            self.node = None
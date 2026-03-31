from prototype.result import Result

class VitapResult(Result):
    def __init__(self, line:str):
        super().__init__()
        self.finalSpecies = None
        terms = line.strip().split('\t')
        taxos = terms[0].split(";")
        for taxo in taxos:
            if (taxo != '-'):
                self.finalSpecies = taxo
                break
        self.lineage_score = float(terms[1])
        self.confidence = terms[2]

    def calcTaxoNode(self):
        if (self.node is None):
            from entity.taxoTree import taxoTree
            if (self.finalSpecies in taxoTree.ICTVTree.nodes):
                self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.finalSpecies)
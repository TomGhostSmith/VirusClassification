from prototype.result import Result
from entity.taxoTree import taxoTree

class CatResult(Result):
    def __init__(self, scores:list[tuple[str, float]]):
        super().__init__()
        self.scores = dict()
        self.finalResult = None
        for NCBIID, score in scores:
            if (NCBIID in taxoTree.viralNCBITree.nodes):
                node = taxoTree.viralNCBITree.nodes[NCBIID]
                self.scores[node.rank] = score
                self.finalResult = node

    def calcTaxoNode(self):
        if (self.node is None):
            if self.finalResult is not None:
                self.node = taxoTree.getTaxoNodeFromNode(NCBINode=self.finalResult)
            else:
                self.node = None
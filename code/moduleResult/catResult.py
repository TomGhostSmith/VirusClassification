from prototype.result import Result

class CatResult(Result):
    def __init__(self, scores:list[tuple[str, float]]):
        super().__init__()
        self.scores = {}
        self.rawResult = scores

    def calcTaxoNode(self):
        if (self.node is None):
            from entity.taxoTree import taxoTree
            finalResult = None
            for NCBIID, score in self.rawResult:
                if (NCBIID in taxoTree.viralNCBITree.nodes):
                    node = taxoTree.viralNCBITree.nodes[NCBIID]
                    self.scores[node.rank] = score
                    finalResult = node
            if finalResult is not None:
                self.node = taxoTree.getTaxoNodeFromNode(NCBINode=finalResult)
                self.score = min(self.scores.values())
            else:
                self.node = None
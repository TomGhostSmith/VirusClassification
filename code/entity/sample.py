from prototype.result import Result
from entity.taxoTree import taxoTree

class Sample():
    # the stdResult is ICTV name
    def __init__(self, query, isATCG, length, stdResult):
        self.query = query
        self.isATCG = isATCG
        self.length = length
        if (stdResult != "no answer"):
            self.stdResult = taxoTree.getTaxoNodeFromICTV(ICTVName=stdResult)
        else:
            self.stdResult = None
        self.results = dict()
        self.info = dict()

    def addResult(self, name:str, result:Result):
        if result is not None:
            result.calcTaxoNode()
        self.results[name] = result

    def addInfo(self, key, value):
        self.info[key] = value
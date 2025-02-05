from prototype.result import Result

class Sample():
    # the stdResult is ICTV name
    def __init__(self, seq):
        self.id = seq.id
        self.isATCG = None
        self.length = len(seq.seq)
        self.seq = seq
        # if (stdResult is not None):
        #     self.stdResult = taxoTree.getTaxoNodeFromICTV(ICTVName=stdResult)
        # else:
        #     self.stdResult = None
        self.results = dict()
        self.info = dict()

    def addResult(self, name:str, result:Result):
        if result is not None:
            result.calcTaxoNode()
        
        self.results[name] = result

    def addInfo(self, key, value):
        self.info[key] = value
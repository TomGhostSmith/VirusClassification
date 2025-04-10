class ProteinSample():
    # the stdResult is ICTV name
    def __init__(self, seq):
        self.id:str = seq.id
        splitID = self.id.rsplit('_', 1)
        self.contigID = splitID[0]
        if (len(splitID) == 2):
            self.index = splitID[1]
        else:
            self.index = 0
        self.length:int = len(seq.seq)
        self.seq = seq
        # if (stdResult is not None):
        #     self.stdResult = taxoTree.getTaxoNodeFromICTV(ICTVName=stdResult)
        # else:
        #     self.stdResult = None
        self.results = dict()
        self.info = dict()
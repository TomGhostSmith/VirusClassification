from prototype.result import Result
from Bio.SeqRecord import SeqRecord
from entity.proteinSample import ProteinSample

class Sample():
    # the stdResult is ICTV name
    def __init__(self, seq:SeqRecord):
        self.id:str = seq.id
        self.isATCG:bool = None
        self.length:int = len(seq.seq)
        self.seq:SeqRecord = seq
        # if (stdResult is not None):
        #     self.stdResult = taxoTree.getTaxoNodeFromICTV(ICTVName=stdResult)
        # else:
        #     self.stdResult = None
        self.results:dict[str, Result] = dict()
        self.info = dict()
        self.proteins:list[ProteinSample] = None

    def addResult(self, name:str, result:Result):
        if result is not None:
            result.calcTaxoNode()
        
        self.results[name] = result

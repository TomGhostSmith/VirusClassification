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
        self.results:dict[str, list[Result]] = {}
        self.info = {}
        self.proteins:list[ProteinSample] = None
        self.cDNAs:list[ProteinSample] = None

    def addResult(self, name:str, results:list[Result]):
        if results is not None:
            results = results[:20]  # at most store 20 results
            for result in results:
                result.calcTaxoNode()
        
        self.results[name] = results

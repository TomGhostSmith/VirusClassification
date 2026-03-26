from prototype.result import Result
from Bio.SeqRecord import SeqRecord

class ProteinSample():
    # the stdResult is ICTV name
    def __init__(self, seq:SeqRecord, head:str=None):
        self.id:str = seq.id
        if (head is None):
            head = seq.description
        self.head = head
        if ("partial=" in head):
            start = head.find("partial=") + 8
            self.partial = head[start:start+2]
        else:
            self.partial = None
        splitID = self.id.rsplit('_', 1)
        self.contigID = splitID[0]
        if (len(splitID) == 2):
            self.index = splitID[1]
        else:
            self.index = 0
        self.length:int = len(seq.seq)
        self.seq:SeqRecord = seq
        self.results:dict[str, list[Result]] = {}
        self.info = {}

    def addResult(self, name:str, results:list[Result]):
        if results is not None:
            results = results[:20]  # at most store 20 results
            for result in results:
                result.calcTaxoNode()
        
        self.results[name] = results
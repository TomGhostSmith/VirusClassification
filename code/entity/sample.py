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
        self.results:dict[str, list[Result]] = {}
        self.info = {}
        self.flags:dict[str, set[str]] = {}
        self.proteins:list[ProteinSample] = None
        self.cDNAs:list[ProteinSample] = None

    def addResult(self, name:str, results:list[Result]):
        if results is not None:
            results = results[:20]  # at most store 20 results
            for result in results:
                result.calcTaxoNode()
        
        self.results[name] = results

    def simplify(self, infos=[], proteinInfos=[]):
        s = Sample(self.seq)
        for k in infos:
            s.info[k] = self.info.get(k)
        s.proteins = []
        for protein in self.proteins:
            p = ProteinSample(protein.seq, protein.head)
            for k in proteinInfos:
                p.info[k] = protein.info.get(k)
            s.proteins.append(p)
        
        return s

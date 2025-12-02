from Bio.SeqRecord import SeqRecord

class ProteinSample():
    # the stdResult is ICTV name
    def __init__(self, seq:SeqRecord, head:str=None):
        self.id:str = seq.id
        if (head is None):
            head = seq.description
        self.head = head
        splitID = self.id.rsplit('_', 1)
        self.contigID = splitID[0]
        if (len(splitID) == 2):
            self.index = splitID[1]
        else:
            self.index = 0
        self.length:int = len(seq.seq)
        self.seq:SeqRecord = seq
        self.results = dict()
        self.info = dict()
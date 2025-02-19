from prototype.result import Result
from entity.taxoTree import taxoTree

class KrakenResult(Result):
    def __init__(self, line:str):
        super().__init__()
        self.finalSpecies = None
        terms = line.strip().split('\t')
        if (terms[0] == 'C'):
            self.finalSpecies = terms[2]  # note: this is kraken ID (NCBI ID)
            self.kmerCounts = dict()
            kmers = terms[4].split(' ')
            for kmer in kmers:
                id, count = kmer.split(':')
                if (id in self.kmerCounts):
                    self.kmerCounts[id] += count
                else:
                    self.kmerCounts[id] = count

    def calcTaxoNode(self):
        self.node = taxoTree.getTaxoNodeFromNCBI(NCBIID=self.finalSpecies)
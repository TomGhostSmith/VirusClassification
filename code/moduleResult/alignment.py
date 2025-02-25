# reconstructed
import re
from entity.taxoTree import taxoTree

class Alignment():
    # def __init__(self, ref, quality, cigar):
    def __init__(self, alignment:str):
        terms = alignment.strip().split('\t')
        if (terms[2] == '*'):
            self.ref = None
        else:
            self.ref = taxoTree.ICTVTree.accession2ID[terms[2]] # the ref is accession, so we convert that to ICTV ID
            self.position = int(terms[3])
            self.quality = int(terms[4])
            self.cigar = terms[5]

            # for extra flags:
            flags = dict()
            for t in terms[11:]:
                key, dtype, value = t.split(':')  # format: key:datatype:value. e.g. de:f:0.00
                if dtype == 'i':
                    value = int(value)
                elif dtype == 'f':
                    value = float(value)
                flags[key] = value

            self.editDis = flags.get('NM')
            self.rawScore = flags.get('ms')
            self.ambiguousPairs = flags.get('nn')
            self.type = flags.get('tp')
            self.errorRate = flags.get('de')

            # calculate coverage & match ratio
            operations = re.findall(r'(\d+)([MIDNSHPX=])', self.cigar)

            queryCoverLength = 0
            refCoverLength = 0
            matchLength = 0

            for length, operation in operations:
                length = int(length)
                if operation == 'M':
                    queryCoverLength += length
                    refCoverLength += length
                    matchLength += length
                elif operation == 'I':
                    queryCoverLength += length
                elif operation == "D":
                    refCoverLength += length
            
            self.queryCoverLength = queryCoverLength
            self.refCoverLength = refCoverLength
            self.matchLength = matchLength

    def betterThan(self, alignment):
        if (self.quality > alignment.quality):
            return True
        else:
            return False
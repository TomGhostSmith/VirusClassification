import re

class Alignment():
    def __init__(self, ref, quality, cigar):
        self.ref = ref  # the ref is supposed to be ICTV ID
        if (ref == '*'):
            self.quality = -1
            self.cigar = None
        else:
            self.quality = quality
            self.cigar = cigar

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
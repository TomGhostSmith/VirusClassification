from module.minimap import Minimap
from moduleResult.alignment import Alignment
from entity.sample import Sample

class MinimapThreshRankModule(Minimap):
    def __init__(self, reference, mode='ont', threads=12, skipComments=True, limitOutputDict=None):
        super().__init__(reference, mode, threads, skipComments)
        self.limitOutputRanks = ["species"]*12
        if (limitOutputDict is not None):
            rankMap = {
                "s": "species",
                "g": "genus",
                "f": "family"
            }
            for k, v in limitOutputDict.items():
                code = self.getCode(k)
                rank = rankMap[v]
                self.limitOutputRanks[code] = rank
        self.code = "".join([t[0] for t in self.limitOutputRanks])
        self.moduleName = f"minimapThRank-ref={self.reference};mode={self.mode};thRank-{self.code}"

    def run(self, samples):
        super().run(samples)
        return [self.getResult(sample) for sample in samples]
        
    def getResult(self, sample:Sample):
        res = super().getResult(sample)
        if (res is not None):
            alignment:Alignment = res.bestAlignment
            key = list()
            if (alignment.quality == 60):
                key.append("60")
            elif (alignment.quality > 0):
                key.append("pos")
            if (alignment.queryCoverLength == sample.length):
                key.append("cm")
            if (len(res.alignments) == 1):
                key.append("sa")
            code = self.getCode(key)
            res.setTargetRank(self.limitOutputRanks[code])
        return res


    def getCode(self, key):
        code = 0b0000
        if ("60" in key):
            code |= 0b1000
        elif ("pos" in key):
            code |= 0b0100
        if ("cm" in key):
            code |= 0b0010
        if ("sa" in key):
            code |= 0b0001
        return code
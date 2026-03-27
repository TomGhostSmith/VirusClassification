import numpy
import multiprocessing
from functools import cmp_to_key

from config import config
from entity.sample import Sample
from module.diamond import Diamond
from moduleResult.plainResult import PlainResult
from moduleResult.diamondAlignment import DiamondAlignment
from moduleResult.blastGenusResult import BlastGenusResult

if multiprocessing.current_process().name == "MainProcess":
    from entity.taxoTree import taxoTree

class DiamondGenus(Diamond):
    def __init__(self, reference, method, threads=multiprocessing.cpu_count(), tool="diamond"):
        if (method not in ["sum", "vote"] and not method.startswith("top")):
            raise ValueError("Unsupported pooling method")
        super().__init__(reference, method, threads, "genus", tool)
        self.moduleName = f'diamondGenus-ref={self.reference};tool={self.tool};method={self.method}'

    
    def getResult(self, sample:Sample, cachedResultFP, withMeta, withProteinMeta, withProteinCandidateMeta, keepProteinRes, withPoolingMeta)->PlainResult:
        # note: result of basename is not available    
        votes:dict[str, int] = dict()

        if (self.method == "sum" or self.method.startswith("top")):
            if (self.method.startswith("top")):
                thresh = int(self.method[3:])
            else:
                thresh = None

        matchedProtein = 0

        for protein in sample.proteins:
            offset, alignmentCount = self.cachedSamples[protein.id]
            cachedResultFP.seek(offset)
            alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline(), self.maxScore) for _ in range(alignmentCount)]
            if (len(alignments) > 0):
                matchedProtein += 1
            alignments = sorted(alignments, key=cmp_to_key(lambda a, b: -1 if a.betterThan(b) else (1 if b.betterThan(a) else 0)))
            candidates:dict[str, BlastGenusResult] = {}
            for alignment in alignments:
                node = taxoTree.getTaxoNodeFromAccession(alignment.refContig).ICTVNode
                genus = None
                for n in reversed(node.path):
                    if n.rank == "genus":
                        genus = n.name
                
                if (genus is None):
                    continue

                if genus not in candidates:
                    candidates[genus] = BlastGenusResult(genus)
                candidates[genus].addAlignment(alignment)

            if (len(candidates) == 0):
                res = None
            else:
                res = list(candidates.values())

            if (keepProteinRes):
                protein.addResult(self.moduleName, res)

            if (withProteinMeta):
                protein.info["proteinAlignments"] = len(alignments)

            if withProteinCandidateMeta and res:
                for r in res:
                    scores = numpy.array([a.similarity for a in r.alignments])
                    lengths = numpy.array([a.length for a in r.alignments])

                    # max score is used as r.score
                    r.info[f"proteinCandidateAvgScore_{self.tool}"] = numpy.average(scores, weights=lengths/sum(lengths))
                    r.info[f"proteinCandidateMinScore_{self.tool}"] = min(scores)
                    r.info[f"proteinCandidateSumScore_{self.tool}"] = sum(scores)
                    r.info[f"proteinCandidateScoreStd_{self.tool}"] = numpy.std(scores)
                    r.info[f"proteinCandidateHits_{self.tool}"] = len(scores)

            if withPoolingMeta and res:
                for r in res:
                    scores = numpy.array([a.similarity for a in r.alignments])
                    r.info["maxScore"] = max(scores)
                    r.info["sumScore"] = sum(scores)
                    r.info["minScore"] = min(scores)

            if (res):
                if (self.method == "vote"):
                    ICTVName = res[0].genus
                    if ICTVName in votes:
                        votes[ICTVName] += 1
                    else:
                        votes[ICTVName] = 1
                elif (self.method == "sum" or self.method.startswith("top")):
                    for r in res[:thresh]:
                        ICTVName = r.genus
                        if ICTVName in votes:
                            votes[ICTVName] += r.score
                        else:
                            votes[ICTVName] = r.score

        results = []
        if len(votes) > 0:
            totalVotes = sum(votes.values())
            vs = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            maxVotes = vs[0][1]
            for name, v in vs:
                r = None
                node = taxoTree.ICTVTree.nodes[name]
                for n in reversed(node.path):
                    if (config.rankLevels[n.rank] <= config.rankLevels[self.threshRank]):
                        r = PlainResult(n.name, score=v/totalVotes)
                        break
                if r is None:
                    r = PlainResult(name, score=v/totalVotes)
                results.append(r)

        else:
            maxVotes = 0
            results = None
            
        if withMeta:
            sample.info[f"matchedProtein_{self.tool}"] = matchedProtein
            sample.info[f"proteinMatchRatio_{self.tool}"] = matchedProtein / len(sample.proteins) if len(sample.proteins) > 0 else 0
            sample.info[f"maxScore_{self.tool}"] = maxVotes / len(sample.proteins) if len(sample.proteins) > 0 else 0
        
        return results
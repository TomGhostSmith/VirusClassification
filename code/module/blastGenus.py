import numpy
import multiprocessing

from config import config
from module.blast import Blast
from moduleResult.blastGenusResult import BlastGenusResult
from moduleResult.blastAlignment import BlastAlignment

if multiprocessing.current_process().name == "MainProcess":
    from entity.taxoTree import taxoTree

class BlastGenus(Blast):
    def __init__(self, reference, threads=multiprocessing.cpu_count(), mode="blastn", evalue=1e-3):
        super().__init__(reference, threads, mode, evalue)
        self.moduleName = f"blastGenus-ref={self.reference};mode={mode};evalue={evalue}"

    def getResult(self, sample, cachedResultFP, withMeta, withCandidateMeta):
        # use baseName to cache the result in the results dict
        offset, alignmentCount = self.cachedSamples[sample.id]
        cachedResultFP.seek(offset)
        alignments:list[BlastAlignment] = [BlastAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
        alignments = sorted(alignments, key=lambda x:x.similarity, reverse=True)
        
        candidates:dict[str, BlastGenusResult] = {}

        nodes = []

        for alignment in alignments:
            if (alignment.ref is not None):
                node = taxoTree.getTaxoNodeFromAccession(alignment.ref).ICTVNode
                nodes.append(node)
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

        if withMeta:
            if (res):
                sample.info["bestBlast"] = res[0].score
                sample.info["blastAlignments"] = len(alignments)
                lca = taxoTree.ICTVTree.findLCA(nodes)
                sample.info["blastAlignmentsLCA"] = config.rankLevels[lca.rank]
            else:
                sample.info["bestBlast"] = 0
                sample.info["blastAlignments"] = 0
                sample.info["blastAlignmentsLCA"] = 0

        if withCandidateMeta and res:
            for r in res:
                scores = numpy.array([a.similarity for a in r.alignments])
                lengths = numpy.array([a.length for a in r.alignments])

                # max score is used as r.score
                r.info["candidateAvgScore_blast"] = numpy.average(scores, weights=lengths/sum(lengths))
                r.info["candidateMinScore_blast"] = min(scores)
                r.info["candidateSumScore_blast"] = sum(scores)
                r.info["candidateScoreStd_blast"] = numpy.std(scores)
                r.info["candidateHits_blast"] = len(scores)

        return res
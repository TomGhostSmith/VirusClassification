# reconstructing
import os
import json
import math
import subprocess
from Bio.Blast import NCBIXML
import multiprocessing

from config import config
from entity.taxoTree import taxoTree
from prototype.module import Module
from moduleResult.blastResult import BlastResult
from moduleResult.blastAlignment import BlastAlignment
from entity.sample import Sample

from utils import IOUtils

class Blast(Module):
    def __init__(self, reference, threads=multiprocessing.cpu_count(), mode="blastn", evalue=1e-3):
        self.reference=reference
        self.threads = threads
        super().__init__(f'blast-ref={self.reference};mode={mode};evalue={evalue}')
        self.baseName = self.moduleName  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.mode = mode
        self.evalue = evalue

        self.cachedSamples:dict[str, tuple[int, int]] = dict()

    def blast(self, samples:list[Sample]):
        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"
        IOUtils.writeSampleFasta(samples, queryFile)
        IOUtils.showInfo(f"Begin blast on {len(samples)} samples")

        command = self.getBlastCommand(queryFile, resultFile)
        subprocess.run(command, shell=True)

        targetFP = open(self.cacheFile, 'at')

        thisName = None
        thisOffset = self.cachedSamples["nextOffset"] if "nextOffset" in self.cachedSamples else 0
        nextOffset = thisOffset
        alignmentCount = 0


        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                sampleName = terms[0]
                if (sampleName != thisName and thisName is not None):
                    # update last sample
                    self.cachedSamples[thisName] = [thisOffset, alignmentCount]
                    thisOffset = nextOffset
                    alignmentCount = 0

                thisName = sampleName
                nextOffset += len(line)
                alignmentCount += 1
                targetFP.write(line)

            
            if (thisName is not None):
                # update last sample
                self.cachedSamples[thisName] = [thisOffset, alignmentCount]
                self.cachedSamples["nextOffset"] = nextOffset
        
        # if there is no alignment, then the query won't show up in the output file
        for sample in samples:
            if sample.id not in self.cachedSamples:
                self.cachedSamples[sample.id] = [0, 0]

        targetFP.close()
        os.remove(resultFile)
        os.remove(queryFile)


    def run(self, samples:list[Sample], **kwargs):
        samplesToRun:list[Sample] = list()

        if (os.path.exists(self.cacheIndex)):
            with open(self.cacheIndex) as fp:
                self.cachedSamples = json.load(fp)  # id: [offset, alignmentCount]

        
            for sample in samples:
                if (sample.id not in self.cachedSamples):
                    samplesToRun.append(sample)
        else:
            samplesToRun = samples
        
        if (len(samplesToRun) > 0):
            self.blast(samplesToRun)
                    
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->BlastResult:
        # use baseName to cache the result in the results dict
        if (self.baseName in sample.results):
            return sample.results[self.baseName]
        
        offset, alignmentCount = self.cachedSamples[sample.id]
        cachedResultFP.seek(offset)
        alignments:list[BlastAlignment] = [BlastAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
        alignments = sorted(alignments, key=lambda x:x.similarity, reverse=True)
        
        results:list[BlastResult] = []
        for alignment in alignments:
            if (alignment.ref is not None):
                r = BlastResult()
                r.addAlignment(alignment)
                r.info["bitscore"] = alignment.bitscore
                r.info["blastSimilarity"] = alignment.similarity
                r.info["blastLogE"] = math.log10(alignment.evalue) if alignment.evalue > 1e-300 else -300
                r.info["bitscoreByLength"] = alignment.bitscore / alignment.length
                r.info["bitscoreByRefCov"] = alignment.bitscore / alignment.refCoverLength
                r.info["bitscoreByQueryCov"] = alignment.bitscore / alignment.queryCoverLength
                r.info["blastRefCov"] = alignment.refCoverLength / sample.length
                r.info["blastQueryCov"] = alignment.queryCoverLength / sample.length
                results.append(r)

        if (len(results) == 0):
            results = None
            sample.info["bestBlast"] = 0
            sample.info["blastAlignments"] = 0
            sample.info["blastAlignmentsLCA"] = 0
        else:
            sample.info["bestBlast"] = results[0].alignment.similarity
            sample.info["blastAlignments"] = len(results)
            nodes = [taxoTree.getTaxoNodeFromAccession(r.alignment.ref).ICTVNode for r in results]
            lca = taxoTree.ICTVTree.findLCA(nodes)
            sample.info["blastAlignmentsLCA"] = config.rankLevels[lca.rank]
        sample.results[self.baseName] = results
        return results
    
    def getBlastCommand(self, queryFile, resultFile):
        # first check if the reference fasta is made a database
        referenceDB = f"{config.cacheResultFolder}/{self.reference}_blastdb"
        dbFiles = [f"{referenceDB}.nhr", f"{referenceDB}.nin", f"{referenceDB}.nsq"]
        dbvalid = True
        for dbFile in dbFiles:
            if not os.path.exists(dbFile):
                dbvalid = False
                break
        if not dbvalid:
            referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
            IOUtils.showInfo(f"Making blast database for {self.reference}")
            subprocess.run(f"makeblastdb -in {referenceFasta} -dbtype nucl -out {referenceDB}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # cline = NcbiblastnCommandline(query=queryFile, db=referenceDB, evalue=1e-3, outfmt=5, out=resultFile)
        # stdout, stderr = cline()
        command = f"blastn -query {queryFile} -db {referenceDB} -evalue {self.evalue} -outfmt 6 -out {resultFile} -num_threads {self.threads} -task {self.mode}"
        return command